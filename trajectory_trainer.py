import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json, glob, os
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from enum import Enum

SEED = 48


def set_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)  # Fix random seed


class SegmentType(Enum):
    TAKEOFF = "takeoff"
    VERTICAL_UP = "vertical_up"
    VERTICAL_DOWN = "vertical_down"
    HORIZONTAL_STRAIGHT = "horizontal_straight"
    HORIZONTAL_DIAGONAL = "horizontal_diagonal"


class GlobalTrajectoryDataset(Dataset):
    """Global dataset containing all segment types"""

    def __init__(self, data_dir="segmented_training_data", max_points=1000):
        self.max_points = max_points
        self.segment_types = [st.value for st in SegmentType]
        self.samples = []

        # Load data for all segment types
        segments_by_type = defaultdict(list)
        for json_file in glob.glob(os.path.join(data_dir, "*_segmented.json")):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for seg in data['segments']:
                    segment_type = seg['segment_info']['segment_type']
                    if segment_type in self.segment_types:
                        segments_by_type[segment_type].append(seg)

        # Process all segment data
        for segment_type, segments_data in segments_by_type.items():
            for segment_data in segments_data:
                segment_info = segment_data['segment_info']
                trajectory_points = segment_data['trajectory_points']

                if len(trajectory_points) < 5:
                    continue

                positions = np.array([[p['position'][0], p['position'][1], p['position'][2]]
                                      for p in trajectory_points])
                velocities = np.array([[p['velocity'][0], p['velocity'][1], p['velocity'][2]]
                                       for p in trajectory_points])

                if len(positions) > max_points:
                    positions = positions[:max_points]
                    velocities = velocities[:max_points]

                sample = {
                    'segment_type': segment_type,
                    'segment_type_idx': self.segment_types.index(segment_type),
                    'start_pos': positions[0],
                    'end_pos': positions[-1],
                    'start_vel': velocities[0],
                    'end_vel': velocities[-1],
                    'duration': segment_info['duration'],
                    'distance': segment_info['distance'],
                    'avg_speed': segment_info['avg_speed'],
                    'num_points': len(positions),
                    'positions': positions,
                    'velocities': velocities
                }
                self.samples.append(sample)

        print(f"Global dataset loaded: {len(self.samples)} samples")

        # Statistics by segment type
        for segment_type in self.segment_types:
            type_samples = [s for s in self.samples if s['segment_type'] == segment_type]
            if type_samples:
                durations = [s['duration'] for s in type_samples]
                distances = [s['distance'] for s in type_samples]
                print(f"  {segment_type}: {len(type_samples)} samples, "
                      f"duration {min(durations):.2f}-{max(durations):.2f}s, "
                      f"distance {min(distances):.4f}-{max(distances):.4f}m")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Pad to maximum length
        positions = np.zeros((self.max_points, 3))
        velocities = np.zeros((self.max_points, 3))

        n = sample['num_points']
        positions[:n] = sample['positions']
        velocities[:n] = sample['velocities']

        return (
            torch.FloatTensor(sample['start_pos']),
            torch.FloatTensor(sample['end_pos']),
            torch.FloatTensor(sample['start_vel']),
            torch.FloatTensor(sample['end_vel']),
            torch.tensor(sample['segment_type_idx'], dtype=torch.long),
            torch.tensor(sample['duration'], dtype=torch.float32),
            torch.tensor(sample['distance'], dtype=torch.float32),
            torch.tensor(sample['avg_speed'], dtype=torch.float32),
            torch.tensor(sample['num_points'], dtype=torch.long),
            torch.FloatTensor(positions),
            torch.FloatTensor(velocities),
            sample['segment_type']  # String for TrajectoryGenerator
        )


class SegmentSpecificDataset(Dataset):
    """Segment-specific dataset for training TrajectoryGenerator"""

    def __init__(self, global_dataset, segment_type):
        self.segment_type = segment_type
        self.samples = []

        # Filter specific segment type from global dataset
        for i in range(len(global_dataset)):
            sample_data = global_dataset[i]
            if sample_data[-1] == segment_type:  # segment_type is the last element
                self.samples.append(sample_data[:-1])  # Exclude segment_type string

        print(f"{segment_type} dataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class GlobalDurationPredictor(nn.Module):
    """Global duration predictor handling all segment types (with direction vector)"""

    def __init__(self, segment_types, hidden_size=128):
        super(GlobalDurationPredictor, self).__init__()

        self.segment_types = segment_types
        self.num_segment_types = len(segment_types)

        # Segment type embedding
        self.segment_embedding = nn.Embedding(self.num_segment_types, 32)

        # Input features:
        # start_pos(3) + end_pos(3) + direction(3) + distance(1) + seg_emb(32)
        geometry_input = 3 + 3 + 3 + 1 + 32
        self.geometry_encoder = nn.Sequential(
            nn.Linear(geometry_input, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )

        # Output heads
        self.duration_head = nn.Linear(hidden_size // 2, 1)
        self.speed_head = nn.Linear(hidden_size // 2, 1)

    def forward(self, start_pos, end_pos, segment_type_idx):
        # Distance
        distance = torch.norm(end_pos - start_pos, dim=1, keepdim=True)

        # Direction vector (normalized)
        direction = end_pos - start_pos
        direction = direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-6)

        # Segment type embedding
        segment_emb = self.segment_embedding(segment_type_idx)

        # Concatenate features
        geometry_features = torch.cat([start_pos, end_pos, direction, distance, segment_emb], dim=1)

        # Encode
        encoded = self.geometry_encoder(geometry_features)

        # Predict temporal features
        duration = torch.sigmoid(self.duration_head(encoded)).squeeze(1) * 14.5 + 0.1  # 0.1–14.6s
        avg_speed = torch.sigmoid(self.speed_head(encoded)).squeeze(1) * 2.0 + 0.001  # 0.001–2.001 m/s

        # Number of sampling points
        num_points = (duration * 100).long() + 1

        return duration, avg_speed, num_points


class SegmentTrajectoryLSTM(nn.Module):
    """Segment-specific trajectory generator (with direction vector)"""

    def __init__(self, segment_types, hidden_size=128, num_layers=2):
        super().__init__()
        self.segment_types = segment_types
        self.num_segment_types = len(segment_types)

        # Segment type embedding
        self.segment_embedding = nn.Embedding(self.num_segment_types, 32)

        # Context input:
        # start_pos(3) + end_pos(3) + start_vel(3) + direction(3) + seg_emb(32)
        context_input = 3 + 3 + 3 + 3 + 32
        self.context_encoder = nn.Sequential(
            nn.Linear(context_input, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # Time encoder
        self.time_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32)
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=32 + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Output layer → 6D [x,y,z,vx,vy,vz]
        self.out = nn.Linear(hidden_size, 6)

    def forward(self, start_pos, end_pos, start_vel, segment_type_idx, num_points, enforce_boundary=True):
        batch_size = start_pos.size(0)
        device = start_pos.device

        # Direction vector (normalized)
        direction = end_pos - start_pos
        direction = direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-6)

        # 1. Context
        seg_emb = self.segment_embedding(segment_type_idx)
        context = torch.cat([start_pos, end_pos, start_vel, direction, seg_emb], dim=1)
        context_vec = self.context_encoder(context)  # [B, H]

        # 2. Time sequence input
        max_len = num_points.max().item()
        t = torch.linspace(0, 1, max_len, device=device).unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1)
        time_emb = self.time_encoder(t)  # [B, T, 32]

        # 3. Concatenate context
        context_seq = context_vec.unsqueeze(1).repeat(1, max_len, 1)  # [B, T, H]
        seq_inputs = torch.cat([time_emb, context_seq], dim=-1)  # [B, T, H+32]

        # 4. LSTM
        outputs, _ = self.lstm(seq_inputs)  # [B, T, H]

        # 5. Output trajectory
        traj = self.out(outputs)  # [B, T, 6]
        positions = traj[:, :, :3]
        velocities = traj[:, :, 3:]

        # 6. Boundary conditions (optional)
        if enforce_boundary:
            for b in range(batch_size):
                n = num_points[b].item()
                positions[b, 0, :] = start_pos[b]
                velocities[b, 0, :] = start_vel[b]
                positions[b, n - 1, :] = end_pos[b]

        return positions, velocities


class CorrectedTrainer:
    """Corrected training system"""

    def __init__(self, data_dir="segmented_training_data"):
        self.segment_types = [st.value for st in SegmentType]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Using device: {self.device}")
        print(f"Segment types: {self.segment_types}")

        # Load global dataset
        print("Loading global dataset...")
        self.global_dataset = GlobalTrajectoryDataset(data_dir)

        # Create global Duration Predictor
        self.global_duration_predictor = GlobalDurationPredictor(self.segment_types).to(self.device)

        # Create Trajectory Generators for each segment type
        self.trajectory_generators = {}
        for segment_type in self.segment_types:
            self.trajectory_generators[segment_type] = SegmentTrajectoryLSTM(self.segment_types).to(self.device)

        self.duration_predictor_trained = False
        self.trajectory_generators_trained = set()

    def train_global_duration_predictor(self, epochs=100, batch_size=32, learning_rate=0.001):
        """Train global Duration Predictor"""
        print(f"\n{'=' * 60}")
        print(f"Training Global Duration Predictor")
        print(f"{'=' * 60}")

        dataloader = DataLoader(self.global_dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.global_duration_predictor.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.7)

        best_loss = float('inf')
        patience = 20
        patience_counter = 0

        print(f"Starting training, dataset size: {len(self.global_dataset)}, batch_size: {batch_size}")

        for epoch in range(epochs):
            self.global_duration_predictor.train()

            total_loss = 0
            duration_errors = []
            speed_errors = []
            points_errors = []

            for batch in dataloader:
                (start_pos, end_pos, start_vel, end_vel, segment_type_idx, duration,
                 distance, avg_speed, num_points, target_positions, target_velocities, _) = batch

                start_pos, end_pos = start_pos.to(self.device), end_pos.to(self.device)
                segment_type_idx = segment_type_idx.to(self.device)
                duration, avg_speed, num_points = duration.to(self.device), avg_speed.to(self.device), num_points.to(
                    self.device)

                optimizer.zero_grad()

                pred_duration, pred_avg_speed, pred_num_points = self.global_duration_predictor(
                    start_pos, end_pos, segment_type_idx
                )

                duration_loss = nn.MSELoss()(pred_duration, duration)
                speed_loss = nn.MSELoss()(pred_avg_speed, avg_speed)
                points_loss = nn.MSELoss()(pred_num_points.float(), num_points.float())

                loss = duration_loss + speed_loss + 0.01 * points_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.global_duration_predictor.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                duration_errors.extend(torch.abs(pred_duration - duration).cpu().detach().numpy())
                speed_errors.extend(torch.abs(pred_avg_speed - avg_speed).cpu().detach().numpy())
                points_errors.extend(torch.abs(pred_num_points.float() - num_points.float()).cpu().detach().numpy())

            avg_loss = total_loss / len(dataloader)
            scheduler.step(avg_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f'Epoch {epoch + 1:3d}/{epochs}: Loss={avg_loss:.6f}, '
                      f'Duration_MAE={np.mean(duration_errors):.4f}s, '
                      f'Speed_MAE={np.mean(speed_errors):.4f}m/s, '
                      f'Points_MAE={np.mean(points_errors):.1f}')

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                self.save_global_duration_predictor(epoch, avg_loss)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        self.duration_predictor_trained = True
        print(f"Global Duration Predictor training completed, best loss: {best_loss:.6f}")
        return True

    def _apply_horizontal_trajectory_correction(self, pred_positions, target_positions, segment_type, num_points):
        """Apply intermediate point correction for horizontal trajectory segments"""
        if segment_type not in ['horizontal_straight', 'horizontal_diagonal']:
            return pred_positions

        corrected_positions = pred_positions.clone()
        batch_size = pred_positions.size(0)

        for b in range(batch_size):
            n = num_points[b].item()
            if n <= 4:  # Skip trajectories that are too short
                continue

            # For horizontal trajectories, apply soft constraints at key intermediate points
            # Select several key intermediate points for correction
            key_indices = []
            if n >= 8:
                # 1/4, 1/2, 3/4 position points
                key_indices = [n // 4, n // 2, 3 * n // 4]
            elif n >= 6:
                # 1/3, 2/3 position points
                key_indices = [n // 3, 2 * n // 3]
            else:
                # Middle point
                key_indices = [n // 2]

            # Apply soft constraints to key points (not hard setting, but weighted average)
            correction_weight = 0.3  # Correction strength, adjustable
            for idx in key_indices:
                if 0 < idx < n - 1:  # Don't correct start and end points
                    corrected_positions[b, idx, :] = (
                            (1 - correction_weight) * pred_positions[b, idx, :] +
                            correction_weight * target_positions[b, idx, :]
                    )

        return corrected_positions

    def train_segment_trajectory_generator(self, segment_type, epochs=150, batch_size=8, learning_rate=0.0005):
        """Train segment-specific Trajectory Generator with enhanced horizontal trajectory intermediate point correction"""
        print(f"\n{'=' * 60}")
        print(f"Training {segment_type} Trajectory Generator")
        if segment_type in ['horizontal_straight', 'horizontal_diagonal']:
            print("Using enhanced horizontal trajectory intermediate point correction")
        print(f"{'=' * 60}")

        if not self.duration_predictor_trained:
            print("Global Duration Predictor not trained, attempting to load...")
            if not self.load_global_duration_predictor():
                print("Unable to load Global Duration Predictor")
                return False

        # Freeze global Duration Predictor
        for param in self.global_duration_predictor.parameters():
            param.requires_grad = False
        print("Global Duration Predictor frozen")

        # Create segment-specific dataset
        segment_dataset = SegmentSpecificDataset(self.global_dataset, segment_type)

        if len(segment_dataset) < 10:
            print(f"Insufficient data for {segment_type}")
            return False

        dataloader = DataLoader(segment_dataset, batch_size=min(batch_size, len(segment_dataset)), shuffle=True)

        # Get corresponding Generator
        generator = self.trajectory_generators[segment_type]

        optimizer = optim.Adam(generator.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.7)

        best_ade = float('inf')
        patience = 30
        patience_counter = 0

        print(f"Starting training, dataset size: {len(segment_dataset)}, batch_size: {min(batch_size, len(segment_dataset))}")

        for epoch in range(epochs):
            self.global_duration_predictor.eval()
            generator.train()

            total_ade = 0
            total_fde = 0
            total_physics_loss = 0
            num_batches = 0

            for batch in dataloader:
                (start_pos, end_pos, start_vel, end_vel, segment_type_idx, duration,
                 distance, avg_speed, num_points, target_positions, target_velocities) = batch

                # Move to device
                start_pos, end_pos = start_pos.to(self.device), end_pos.to(self.device)
                start_vel, end_vel = start_vel.to(self.device), end_vel.to(self.device)
                segment_type_idx = segment_type_idx.to(self.device)
                duration, distance, avg_speed = duration.to(self.device), distance.to(self.device), avg_speed.to(
                    self.device)
                num_points = num_points.to(self.device)
                target_positions, target_velocities = target_positions.to(self.device), target_velocities.to(
                    self.device)

                optimizer.zero_grad()

                # Use fixed global Duration Predictor for prediction
                with torch.no_grad():
                    pred_duration, pred_avg_speed, pred_num_points = self.global_duration_predictor(
                        start_pos, end_pos, segment_type_idx
                    )

                # Train Trajectory Generator
                pred_positions, pred_velocities = generator(
                    start_pos, end_pos, start_vel, segment_type_idx, num_points,
                    enforce_boundary=False
                )

                # [NEW] Apply intermediate point correction for horizontal trajectory segments
                if segment_type in ['horizontal_straight', 'horizontal_diagonal']:
                    pred_positions = self._apply_horizontal_trajectory_correction(
                        pred_positions, target_positions, segment_type, num_points
                    )

                # Calculate loss
                batch_size = start_pos.size(0)
                batch_ade = 0
                batch_fde = 0
                physics_loss = 0

                for b in range(batch_size):
                    n = num_points[b].item()

                    # ADE/FDE
                    errors = torch.norm(pred_positions[b, :n, :] - target_positions[b, :n, :], dim=1)
                    batch_ade += torch.mean(errors).item()
                    batch_fde += errors[-1].item()

                    # Physics consistency loss
                    if n > 1:
                        pred_vel_from_pos = (pred_positions[b, 1:n, :] - pred_positions[b, :n - 1, :]) / 0.01
                        physics_loss += nn.MSELoss()(pred_vel_from_pos, pred_velocities[b, :n - 1, :])

                batch_ade /= batch_size
                batch_fde /= batch_size
                physics_loss /= batch_size

                # Trajectory reconstruction loss
                pos_loss = 0
                vel_loss = 0

                for b in range(batch_size):
                    n = num_points[b].item()  # Actual trajectory points for this sample
                    pos_loss += nn.MSELoss()(pred_positions[b, :n, :], target_positions[b, :n, :])
                    vel_loss += nn.MSELoss()(pred_velocities[b, :n, :], target_velocities[b, :n, :])

                # Full batch acceleration
                pred_acc = pred_velocities[:, 1:, :] - pred_velocities[:, :-1, :]
                acc_loss = torch.mean(torch.abs(pred_acc))

                pos_loss /= batch_size
                vel_loss /= batch_size

                # Boundary loss
                boundary_loss = 0
                for b in range(batch_size):
                    n = num_points[b].item()
                    boundary_loss += (nn.MSELoss()(pred_positions[b, 0, :], start_pos[b]) +
                                      nn.MSELoss()(pred_positions[b, n - 1, :], end_pos[b]) +
                                      nn.MSELoss()(pred_velocities[b, 0, :], start_vel[b]) +
                                      nn.MSELoss()(pred_velocities[b, n - 1, :], end_vel[b]))
                boundary_loss /= batch_size

                # [MODIFIED] Increase boundary loss weight for horizontal trajectory segments
                if segment_type in ['horizontal_straight', 'horizontal_diagonal']:
                    total_loss = pos_loss + vel_loss + 5.0 * physics_loss + 15.0 * boundary_loss + 0.02 * acc_loss  # Increase boundary loss weight
                else:
                    total_loss = pos_loss + vel_loss + 5.0 * physics_loss + 10.0 * boundary_loss + 0.02 * acc_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                optimizer.step()

                total_ade += batch_ade
                total_fde += batch_fde
                total_physics_loss += physics_loss.item()
                num_batches += 1

            avg_ade = total_ade / num_batches
            avg_fde = total_fde / num_batches
            avg_physics = total_physics_loss / num_batches

            scheduler.step(avg_ade)

            if (epoch + 1) % 15 == 0 or epoch == 0:
                print(f'Epoch {epoch + 1:3d}/{epochs}: '
                      f'ADE={avg_ade:.6f}m, '
                      f'FDE={avg_fde:.6f}m, '
                      f'Physics={avg_physics:.6f}')

            if avg_ade < best_ade:
                best_ade = avg_ade
                patience_counter = 0
                self.save_segment_generator(segment_type, epoch, avg_ade)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        # Unfreeze global Duration Predictor
        for param in self.global_duration_predictor.parameters():
            param.requires_grad = True

        self.trajectory_generators_trained.add(segment_type)
        print(f"{segment_type} Trajectory Generator training completed, best ADE: {best_ade:.6f}m")
        return True

    def train_all_segment_generators(self):
        """Train all segment-specific Trajectory Generators"""
        if not self.duration_predictor_trained:
            print("Please train Global Duration Predictor first")
            return

        results = {}
        for segment_type in self.segment_types:
            try:
                success = self.train_segment_trajectory_generator(segment_type)
                results[segment_type] = "Success" if success else "Failed"
            except Exception as e:
                print(f"{segment_type} training exception: {e}")
                results[segment_type] = f"Exception: {e}"

        print(f"\n{'=' * 60}")
        print("All Trajectory Generator training results:")
        print(f"{'=' * 60}")
        for segment_type, result in results.items():
            print(f"{segment_type:20s}: {result}")

        return results

    def save_global_duration_predictor(self, epoch, loss):
        os.makedirs("models", exist_ok=True)

        torch.save({
            'model_state_dict': self.global_duration_predictor.state_dict(),
            'segment_types': self.segment_types,
            'epoch': epoch,
            'seed': SEED,
            'loss': loss
        }, "models/global_duration_predictor.pth")

    def save_segment_generator(self, segment_type, epoch, ade):
        os.makedirs("models/segment_generators", exist_ok=True)

        torch.save({
            'model_state_dict': self.trajectory_generators[segment_type].state_dict(),
            'segment_type': segment_type,
            'epoch': epoch,
            'seed': SEED,
            'ade': ade
        }, f"models/segment_generators/{segment_type}_generator.pth")

    def load_global_duration_predictor(self):
        model_path = "models/global_duration_predictor.pth"

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.global_duration_predictor.load_state_dict(checkpoint['model_state_dict'])
            self.duration_predictor_trained = True
            print("Global Duration Predictor loaded successfully")
            return True
        else:
            print(f"Model file not found: {model_path}")
            return False

    def load_segment_generator(self, segment_type):
        model_path = f"models/segment_generators/{segment_type}_generator.pth"

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.trajectory_generators[segment_type].load_state_dict(checkpoint['model_state_dict'])
            self.trajectory_generators_trained.add(segment_type)
            print(f"{segment_type} Generator loaded successfully")
            return True
        else:
            print(f"Model file not found: {model_path}")
            return False


def main():
    print("Architecture: 1 Global Duration Predictor + 5 Specialized Trajectory Generators")
    print("Horizontal trajectory enhancement: Added intermediate point correction functionality")
    print("=" * 80)

    trainer = CorrectedTrainer()

    print("Training options:")
    print("1. Train Global Duration Predictor")
    print("2. Train single segment type Trajectory Generator")
    print("3. Train all Trajectory Generators")
    print("4. Complete training (Duration Predictor + all Trajectory Generators)")
    print("5. Train only horizontal trajectory segments (horizontal_straight + horizontal_diagonal)")

    choice = input("Please select (1-5): ").strip()

    if choice == "1":
        trainer.train_global_duration_predictor()

    elif choice == "2":
        print("\nAvailable segment types:")
        for i, seg_type in enumerate(trainer.segment_types):
            print(f"{i + 1}. {seg_type}")

        seg_choice = input("Select segment type number: ").strip()
        try:
            seg_idx = int(seg_choice) - 1
            segment_type = trainer.segment_types[seg_idx]
            trainer.train_segment_trajectory_generator(segment_type)
        except (ValueError, IndexError):
            print("Invalid selection")

    elif choice == "3":
        trainer.train_all_segment_generators()

    elif choice == "4":
        # Complete training
        print("Starting complete training process...")

        # Phase 1: Train Global Duration Predictor
        success = trainer.train_global_duration_predictor()
        if not success:
            print("Global Duration Predictor training failed, terminating training")
            return

        # Phase 2: Train all Trajectory Generators
        trainer.train_all_segment_generators()

        print("\nComplete training process finished!")

    elif choice == "5":
        # Train only horizontal trajectory segments
        print("Starting horizontal trajectory segment training...")

        horizontal_types = ['horizontal_straight', 'horizontal_diagonal']
        results = {}

        for segment_type in horizontal_types:
            try:
                success = trainer.train_segment_trajectory_generator(segment_type)
                results[segment_type] = "Success" if success else "Failed"
            except Exception as e:
                print(f"{segment_type} training exception: {e}")
                results[segment_type] = f"Exception: {e}"

        print(f"\n{'=' * 60}")
        print("Horizontal trajectory segment training results:")
        print(f"{'=' * 60}")
        for segment_type, result in results.items():
            print(f"{segment_type:20s}: {result}")

    else:
        print("Invalid selection")


if __name__ == "__main__":
    main()