import os, json, glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from enum import Enum
from collections import defaultdict


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



def evaluate_segment_ade(pred_seg, gt_seg):
    n = min(len(pred_seg['positions']), len(gt_seg['positions']))
    p = pred_seg['positions'][:n]
    g = gt_seg['positions'][:n]
    return float(np.mean(np.linalg.norm(p - g, axis=1)))


def evaluate_all_files_by_segment_type(data_dir="segmented_training_data"):
    """Evaluate ADE for all files grouped by segment type"""
    infer = TrajectoryInferenceSystem(models_dir="models")
    infer.load_models()

    # Collect all errors by segment type
    segment_errors = defaultdict(list)

    json_files = glob.glob(os.path.join(data_dir, "*_segmented.json"))
    print(f"Processing {len(json_files)} files for segment-wise ADE evaluation...")

    for json_file in json_files:
        try:
            # Load ground truth
            _, _, seg_defs, gt_segments = load_ground_truth_from_segmented_json(json_file)

            # Generate predictions
            _, _, _, pred_segments_global = infer.predict_full_trajectory(seg_defs)

            # Calculate ADE for each segment
            for pred_seg, gt_seg in zip(pred_segments_global, gt_segments):
                seg_type = gt_seg['type']
                ade = evaluate_segment_ade(pred_seg, gt_seg)
                segment_errors[seg_type].append(ade)

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    # Calculate average ADE for each segment type
    avg_segment_ade = {}
    for seg_type, errors in segment_errors.items():
        avg_segment_ade[seg_type] = np.mean(errors)
        print(f"{seg_type}: {len(errors)} segments, ADE = {avg_segment_ade[seg_type]:.4f} m")

    return avg_segment_ade


def plot_segment_ade_bar_chart(avg_segment_ade, save_path="segment_ade_comparison.png"):
    """Plot bar chart with ADE values displayed on top of bars"""
    plt.figure(figsize=(12, 6))

    segment_types = list(avg_segment_ade.keys())
    ade_values = list(avg_segment_ade.values())

    bars = plt.bar(range(len(segment_types)), ade_values, color='steelblue', alpha=0.8)

    # Add value labels on top of bars
    for i, (bar, value) in enumerate(zip(bars, ade_values)):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(ade_values) * 0.01,
                 f'{value:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.xticks(range(len(segment_types)), segment_types, rotation=45, ha='right')
    plt.ylabel('ADE (m)', fontsize=12)
    plt.title('Average Displacement Error by Segment Type', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    # Add some padding at the top
    plt.ylim(0, max(ade_values) * 1.15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Segment ADE comparison saved: {save_path}")



def main():
    print("Evaluating ADE across all files by segment type...")
    avg_segment_ade = evaluate_all_files_by_segment_type("segmented_training_data")

    print("\nAverage ADE by segment type:")
    for seg_type, ade in avg_segment_ade.items():
        print(f"  {seg_type}: {ade:.4f} m")

    # Plot bar chart
    plot_segment_ade_bar_chart(avg_segment_ade)


if __name__ == "__main__":
    main()