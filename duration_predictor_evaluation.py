import os, json, glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
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


def load_ground_truth_from_segmented_json(json_path):
    """Load ground truth data from segmented JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    segment_samples = []
    for seg in data['segments']:
        seg_type = seg['segment_info']['segment_type']
        pts = seg['trajectory_points']
        if len(pts) < 2:
            continue

        pos = np.array([p['position'] for p in pts], dtype=np.float32)
        duration = seg['segment_info']['duration']

        segment_samples.append({
            'start_pos': pos[0].tolist(),
            'end_pos': pos[-1].tolist(),
            'segment_type': seg_type,
            'duration': duration
        })

    return segment_samples


def evaluate_duration_by_segment_type(data_dir="segmented_training_data"):
    """Evaluate time prediction for all files, grouped by segment type"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    segment_types = [st.value for st in SegmentType]

    # Load model
    model_path = "models/global_duration_predictor.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = GlobalDurationPredictor(segment_types)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()

    # Collect errors by segment type
    segment_errors = defaultdict(list)

    json_files = glob.glob(os.path.join(data_dir, "*_segmented.json"))
    print(f"Processing {len(json_files)} files for time prediction evaluation...")

    with torch.no_grad():
        for json_file in json_files:
            try:
                # Load ground truth data
                segments = load_ground_truth_from_segmented_json(json_file)

                for seg in segments:
                    if seg['segment_type'] not in segment_types:
                        continue

                    # Prepare inputs
                    start_pos = torch.tensor(seg['start_pos'], dtype=torch.float32, device=device).unsqueeze(0)
                    end_pos = torch.tensor(seg['end_pos'], dtype=torch.float32, device=device).unsqueeze(0)
                    seg_type_idx = torch.tensor([segment_types.index(seg['segment_type'])], device=device)

                    # Predict duration
                    pred_duration, _, _ = model(start_pos, end_pos, seg_type_idx)
                    pred_duration = pred_duration.cpu().numpy()[0]

                    # Calculate duration error
                    duration_error = abs(pred_duration - seg['duration'])
                    segment_errors[seg['segment_type']].append(duration_error)

            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue

    # Calculate average duration error by segment type
    avg_duration_errors = {}
    for seg_type, errors in segment_errors.items():
        avg_duration_errors[seg_type] = np.mean(errors)
        print(f"{seg_type}: {len(errors)} segments, Duration MAE = {avg_duration_errors[seg_type]:.4f} seconds")

    return avg_duration_errors


def plot_duration_mae_bar_chart(avg_duration_errors, save_path="duration_mae_by_segment.png"):
    """Plot duration prediction MAE bar chart by segment type"""
    plt.figure(figsize=(12, 6))

    segment_types = list(avg_duration_errors.keys())
    mae_values = list(avg_duration_errors.values())

    bars = plt.bar(range(len(segment_types)), mae_values, color='steelblue', alpha=0.8)

    # Add value labels on top of bars
    for i, (bar, value) in enumerate(zip(bars, mae_values)):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(mae_values) * 0.01,
                 f'{value:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.xticks(range(len(segment_types)), segment_types, rotation=45, ha='right')
    plt.ylabel('Duration MAE (s)', fontsize=12)
    plt.title('Duration Prediction Error by Segment Type', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    # Add some padding at the top
    plt.ylim(0, max(mae_values) * 1.15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Duration prediction MAE comparison saved: {save_path}")


def main():
    print("Evaluating duration prediction MAE by segment type...")
    avg_duration_errors = evaluate_duration_by_segment_type("segmented_training_data")

    print("\nDuration prediction MAE by segment type:")
    for seg_type, mae in avg_duration_errors.items():
        print(f"  {seg_type}: {mae:.4f} seconds")

    # Plot bar chart
    plot_duration_mae_bar_chart(avg_duration_errors)


if __name__ == "__main__":
    main()