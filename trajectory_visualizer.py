import os, json
import numpy as np
import torch
import matplotlib.pyplot as plt
from enum import Enum



plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False



class SegmentType(Enum):
    TAKEOFF = "takeoff"
    VERTICAL_UP = "vertical_up"
    VERTICAL_DOWN = "vertical_down"
    HORIZONTAL_STRAIGHT = "horizontal_straight"
    HORIZONTAL_DIAGONAL = "horizontal_diagonal"



class GlobalDurationPredictor(torch.nn.Module):
    def __init__(self, segment_types, hidden_size=128):
        super().__init__()
        self.segment_types = segment_types
        self.num_segment_types = len(segment_types)
        self.segment_embedding = torch.nn.Embedding(self.num_segment_types, 32)
        geometry_input = 3 + 3 + 3 + 1 + 32
        self.geometry_encoder = torch.nn.Sequential(
            torch.nn.Linear(geometry_input, hidden_size), torch.nn.ReLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size, hidden_size // 2), torch.nn.ReLU()
        )
        self.duration_head = torch.nn.Linear(hidden_size // 2, 1)
        self.speed_head = torch.nn.Linear(hidden_size // 2, 1)

    def forward(self, start_pos, end_pos, segment_type_idx):
        distance = torch.norm(end_pos - start_pos, dim=1, keepdim=True)
        direction = end_pos - start_pos
        direction = direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-6)
        seg_emb = self.segment_embedding(segment_type_idx)
        feat = torch.cat([start_pos, end_pos, direction, distance, seg_emb], dim=1)
        enc = self.geometry_encoder(feat)
        duration = torch.sigmoid(self.duration_head(enc)).squeeze(1) * 15.0 + 0.1
        avg_speed = torch.sigmoid(self.speed_head(enc)).squeeze(1) * 2.0 + 0.001
        num_points = (duration * 100).long() + 1
        return duration, avg_speed, num_points


class SegmentTrajectoryLSTM(torch.nn.Module):
    def __init__(self, segment_types, hidden_size=128, num_layers=2):
        super().__init__()
        self.segment_types = segment_types
        self.num_segment_types = len(segment_types)
        self.segment_embedding = torch.nn.Embedding(self.num_segment_types, 32)
        context_input = 3 + 3 + 3 + 3 + 32
        self.context_encoder = torch.nn.Sequential(
            torch.nn.Linear(context_input, hidden_size), torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size)
        )
        self.time_encoder = torch.nn.Sequential(
            torch.nn.Linear(1, 32), torch.nn.Tanh(), torch.nn.Linear(32, 32)
        )
        self.lstm = torch.nn.LSTM(input_size=32 + hidden_size,
                                  hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.out = torch.nn.Linear(hidden_size, 6)

    def forward(self, start_pos, end_pos, start_vel, segment_type_idx, num_points, enforce_boundary=True):
        B = start_pos.size(0);
        device = start_pos.device
        direction = end_pos - start_pos
        direction = direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-6)
        seg_emb = self.segment_embedding(segment_type_idx)
        context = torch.cat([start_pos, end_pos, start_vel, direction, seg_emb], dim=1)
        ctx = self.context_encoder(context)
        T = num_points.max().item()
        t = torch.linspace(0, 1, T, device=device).unsqueeze(0).repeat(B, 1).unsqueeze(-1)
        time_emb = self.time_encoder(t)
        seq_in = torch.cat([time_emb, ctx.unsqueeze(1).repeat(1, T, 1)], dim=-1)
        h, _ = self.lstm(seq_in)
        traj = self.out(h)
        pos, vel = traj[:, :, :3], traj[:, :, 3:]

        if enforce_boundary:
            for b in range(B):
                n = num_points[b].item()
                pos[b, 0, :] = start_pos[b]
                vel[b, 0, :] = start_vel[b]
                pos[b, n - 1, :] = end_pos[b]
        return pos, vel



class TrajectoryInferenceSystem:
    def __init__(self, models_dir="models"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.segment_types = [st.value for st in SegmentType]
        self.models_dir = models_dir
        self.global_duration_predictor = None
        self.trajectory_generators = {}

    def load_models(self):
        dur_path = os.path.join(self.models_dir, "global_duration_predictor.pth")
        if not os.path.exists(dur_path):
            raise FileNotFoundError("Model not found: " + dur_path)
        self.global_duration_predictor = GlobalDurationPredictor(self.segment_types)
        ckpt = torch.load(dur_path, map_location=self.device, weights_only=False)
        self.global_duration_predictor.load_state_dict(ckpt['model_state_dict'])
        self.global_duration_predictor.to(self.device).eval()

        gen_dir = os.path.join(self.models_dir, "segment_generators")
        for seg in self.segment_types:
            p = os.path.join(gen_dir, f"{seg}_generator.pth")
            if os.path.exists(p):
                g = SegmentTrajectoryLSTM(self.segment_types)
                ck = torch.load(p, map_location=self.device, weights_only=False)
                g.load_state_dict(ck['model_state_dict'])
                g.to(self.device).eval()
                self.trajectory_generators[seg] = g

    def predict_single_segment(self, start_pos, end_pos, start_vel, segment_type):
        if segment_type not in self.trajectory_generators:
            raise ValueError(f"Generator not loaded for {segment_type}")
        sp = torch.tensor(start_pos, dtype=torch.float32, device=self.device).unsqueeze(0)
        ep = torch.tensor(end_pos, dtype=torch.float32, device=self.device).unsqueeze(0)
        sv = torch.tensor(start_vel, dtype=torch.float32, device=self.device).unsqueeze(0)
        idx = torch.tensor([self.segment_types.index(segment_type)], device=self.device)
        with torch.no_grad():
            dur, _, npts = self.global_duration_predictor(sp, ep, idx)
            pos, vel = self.trajectory_generators[segment_type](sp, ep, sv, idx, npts, enforce_boundary=True)
        n = int(npts.item());
        d = float(dur.item())
        t_local = np.linspace(0.0, d, n)
        return dict(
            segment_type=segment_type, duration=d, num_points=n,
            positions=pos[0, :n, :].cpu().numpy(), velocities=vel[0, :n, :].cpu().numpy(),
            times=t_local
        )

    def predict_full_trajectory(self, seg_defs):
        seg_preds = []
        start_vel = [0.0, 0.0, 0.0]
        for s in seg_defs:
            pred = self.predict_single_segment(s['start_pos'], s['end_pos'], start_vel, s['segment_type'])
            seg_preds.append(pred)
            start_vel = pred['velocities'][-1]

        positions, times, pred_segments_global = [], [], []
        t0 = 0.0
        for k, seg in enumerate(seg_preds):
            pos_local = seg['positions']
            t_abs = seg['times'] + t0
            pred_segments_global.append(dict(
                type=seg['segment_type'],
                positions=pos_local.copy(),
                times=t_abs.copy()
            ))
            if k > 0: pos_local = pos_local[1:]; t_abs = t_abs[1:]
            positions.append(pos_local);
            times.append(t_abs)
            t0 = pred_segments_global[-1]['times'][-1]

        return np.vstack(positions), np.concatenate(times), seg_preds, pred_segments_global



def load_ground_truth_from_segmented_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_pos, all_time, seg_defs, gt_segments = [], [], [], []
    t0, first_point = 0.0, True

    for seg in data['segments']:
        seg_type = seg['segment_info']['segment_type']
        pts = seg['trajectory_points']
        if len(pts) < 2:
            continue
        pos = np.array([p['position'] for p in pts], dtype=np.float32)
        t_raw = np.array([p['time'] for p in pts], dtype=np.float64)
        t_rel = t_raw - t_raw[0]
        t_abs = t_rel + t0

        seg_defs.append(dict(start_pos=pos[0].tolist(), end_pos=pos[-1].tolist(), segment_type=seg_type))
        gt_segments.append(dict(type=seg_type, positions=pos.copy(), times=t_abs.copy()))

        if first_point:
            all_pos.append(pos);
            all_time.append(t_abs);
            first_point = False
        else:
            all_pos.append(pos[1:]);
            all_time.append(t_abs[1:])
        t0 = t_abs[-1]

    return np.vstack(all_pos), np.concatenate(all_time), seg_defs, gt_segments



def _set_3d_equal(ax, X, Y, Z):
    """Make 3D axes equal scale"""
    x_range = X.max() - X.min()
    y_range = Y.max() - Y.min()
    z_range = Z.max() - Z.min()
    r = max(x_range, y_range, z_range) * 0.5
    x_mid = (X.max() + X.min()) / 2
    y_mid = (Y.max() + Y.min()) / 2
    z_mid = (Z.max() + Z.min()) / 2
    ax.set_xlim(x_mid - r, x_mid + r)
    ax.set_ylim(y_mid - r, y_mid + r)
    ax.set_zlim(z_mid - r, z_mid + r)
    ax.set_box_aspect([1, 1, 1])


def align_trajectories_by_length(pred_pos, true_pos):
    """Align trajectories by taking the minimum length"""
    n = min(len(pred_pos), len(true_pos))
    return pred_pos[:n], true_pos[:n]


def calculate_ade(pred_pos, true_pos):
    """Calculate Average Displacement Error"""
    pred_aligned, true_aligned = align_trajectories_by_length(pred_pos, true_pos)
    return np.mean(np.linalg.norm(pred_aligned - true_aligned, axis=1))



def plot_vertical_trajectory_comparison(pred_pos, pred_t, true_pos, true_t, mission_name, save_path=None):
    """Visualization for vertical flight trajectories"""
    # Align trajectories
    pred_aligned, true_aligned = align_trajectories_by_length(pred_pos, true_pos)
    ade = calculate_ade(pred_pos, true_pos)

    fig = plt.figure(figsize=(16, 6))

    # 3D Trajectory Comparison
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], label='Predicted', linewidth=2.5, color='blue')
    ax1.plot(true_pos[:, 0], true_pos[:, 1], true_pos[:, 2], '--', label='Ground Truth', linewidth=2.5, color='red')
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_zlabel('Z (m)', fontsize=12)
    ax1.set_title('3D Trajectory Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    _set_3d_equal(ax1, np.r_[pred_pos[:, 0], true_pos[:, 0]],
                  np.r_[pred_pos[:, 1], true_pos[:, 1]],
                  np.r_[pred_pos[:, 2], true_pos[:, 2]])

    # Height vs Time (Aligned)
    ax2 = fig.add_subplot(122)
    time_indices = np.arange(len(pred_aligned))
    ax2.plot(time_indices, pred_aligned[:, 2], label='Predicted', linewidth=2.5, color='blue')
    ax2.plot(time_indices, true_aligned[:, 2], '--', label='Ground Truth', linewidth=2.5, color='red')
    ax2.set_xlabel('Time Index', fontsize=12)
    ax2.set_ylabel('Height (m)', fontsize=12)
    ax2.set_title(f'Height vs Time Comparison\nADE: {ade:.4f} m', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'{mission_name} - Vertical Flight Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Vertical trajectory comparison saved: {save_path}")
    plt.show()


def plot_horizontal_trajectory_comparison(pred_pos, pred_t, true_pos, true_t, mission_name, save_path=None):
    """Visualization for horizontal flight trajectories (square/triangle)"""
    # Align trajectories
    pred_aligned, true_aligned = align_trajectories_by_length(pred_pos, true_pos)
    ade = calculate_ade(pred_pos, true_pos)

    fig = plt.figure(figsize=(16, 6))

    # 3D Trajectory Comparison
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], label='Predicted', linewidth=2.5, color='blue')
    ax1.plot(true_pos[:, 0], true_pos[:, 1], true_pos[:, 2], '--', label='Ground Truth', linewidth=2.5, color='red')
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_zlabel('Z (m)', fontsize=12)
    ax1.set_title('3D Trajectory Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    _set_3d_equal(ax1, np.r_[pred_pos[:, 0], true_pos[:, 0]],
                  np.r_[pred_pos[:, 1], true_pos[:, 1]],
                  np.r_[pred_pos[:, 2], true_pos[:, 2]])

    # X-Y Plane Comparison (Aligned)
    ax2 = fig.add_subplot(122)
    ax2.plot(pred_aligned[:, 0], pred_aligned[:, 1], label='Predicted', linewidth=2.5, color='blue')
    ax2.plot(true_aligned[:, 0], true_aligned[:, 1], '--', label='Ground Truth', linewidth=2.5, color='red')
    ax2.set_xlabel('X (m)', fontsize=12)
    ax2.set_ylabel('Y (m)', fontsize=12)
    ax2.set_title(f'X-Y Plane Comparison\nADE: {ade:.4f} m', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    plt.suptitle(f'{mission_name} - Horizontal Flight Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Horizontal trajectory comparison saved: {save_path}")
    plt.show()


def determine_mission_type(mission_name):
    """Determine if mission is vertical or horizontal based on name"""
    mission_name_lower = mission_name.lower()
    if 'vertical' in mission_name_lower:
        return 'vertical'
    elif 'square' in mission_name_lower or 'triangle' in mission_name_lower:
        return 'horizontal'
    else:
        return 'unknown'


def visualize_trajectory_comparison(json_path, models_dir="models"):
    """Main function to visualize trajectory comparison"""
    # Load ground truth
    true_pos, true_t, seg_defs, gt_segments = load_ground_truth_from_segmented_json(json_path)

    # Get mission name from file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    mission_name = data.get('file_info', {}).get('original_filename', os.path.basename(json_path))

    # Generate predictions
    infer = TrajectoryInferenceSystem(models_dir=models_dir)
    infer.load_models()
    pred_pos, pred_t, _, pred_segments_global = infer.predict_full_trajectory(seg_defs)

    # Determine mission type and visualize accordingly
    mission_type = determine_mission_type(mission_name)

    base_filename = os.path.splitext(os.path.basename(json_path))[0]

    if mission_type == 'vertical':
        save_path = f"{base_filename}_vertical_comparison.png"
        plot_vertical_trajectory_comparison(pred_pos, pred_t, true_pos, true_t, mission_name, save_path)
    elif mission_type == 'horizontal':
        save_path = f"{base_filename}_horizontal_comparison.png"
        plot_horizontal_trajectory_comparison(pred_pos, pred_t, true_pos, true_t, mission_name, save_path)
    else:
        print(f"Unknown mission type for {mission_name}, using horizontal visualization")
        save_path = f"{base_filename}_unknown_comparison.png"
        plot_horizontal_trajectory_comparison(pred_pos, pred_t, true_pos, true_t, mission_name, save_path)

    # Print summary
    ade = calculate_ade(pred_pos, true_pos)
    print(f"\nTrajectory Comparison Summary:")
    print(f"Mission: {mission_name}")
    print(f"Mission Type: {mission_type}")
    print(f"Overall ADE: {ade:.4f} m")
    print(f"Predicted trajectory points: {len(pred_pos)}")
    print(f"Ground truth trajectory points: {len(true_pos)}")



def main():
    json_path = r"C:\Users\51539\Desktop\Drone_Trajectory_LSTM\segmented_training_data\triangle_med_20250905_162108_segmented.json"

    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        print("Please update the json_path variable with the correct file path.")
        return

    print(f"Visualizing trajectory comparison for: {os.path.basename(json_path)}")
    visualize_trajectory_comparison(json_path, models_dir="models")


if __name__ == "__main__":
    main()