import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Set matplotlib backend
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class SegmentType(Enum):
    TAKEOFF = "takeoff"
    VERTICAL_UP = "vertical_up"
    VERTICAL_DOWN = "vertical_down"
    HORIZONTAL_STRAIGHT = "horizontal_straight"
    HORIZONTAL_DIAGONAL = "horizontal_diagonal"


@dataclass
class TrajectorySegment:
    segment_id: int
    start_time: float
    end_time: float
    start_idx: int
    end_idx: int
    segment_type: SegmentType
    start_waypoint: Tuple[float, float, float]
    end_waypoint: Tuple[float, float, float]
    duration: float
    distance: float
    avg_speed: float
    states: List[Dict]

    def to_dict(self) -> Dict:
        """Convert TrajectorySegment to dictionary for JSON serialization"""
        return {
            'segment_id': self.segment_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'start_idx': self.start_idx,
            'end_idx': self.end_idx,
            'segment_type': self.segment_type.value,
            'start_waypoint': self.start_waypoint,
            'end_waypoint': self.end_waypoint,
            'duration': self.duration,
            'distance': self.distance,
            'avg_speed': self.avg_speed,
            'num_states': len(self.states)
        }


class DroneTrajectorySegmenter:
    def __init__(self):
        """Initialize the trajectory segmenter"""
        pass

    def load_json_data(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load JSON data file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def calculate_distance(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    def calculate_speed(self, velocity: List[float]) -> float:
        """Calculate speed magnitude"""
        return np.sqrt(sum(v ** 2 for v in velocity))

    def is_waypoint_reached(self, position: List[float], waypoint: Tuple[float, float, float],
                            velocity: List[float], pos_threshold: float = 0.08,
                            vel_threshold: float = 0.2) -> bool:
        """Check if waypoint is reached: position within 0.08m and speed below 0.3m/s"""
        distance = self.calculate_distance(position, waypoint)
        speed = self.calculate_speed(velocity)
        return distance <= pos_threshold and speed <= vel_threshold

    def classify_segment_type(self, start_waypoint: Tuple[float, float, float],
                              end_waypoint: Tuple[float, float, float],
                              mission_name: str) -> SegmentType:
        """Classify segment type based on waypoints and mission type"""
        dz = end_waypoint[2] - start_waypoint[2]
        dx = end_waypoint[0] - start_waypoint[0]
        dy = end_waypoint[1] - start_waypoint[1]
        horizontal_distance = np.sqrt(dx * dx + dy * dy)

        # Vertical movement (significant height change with minimal horizontal movement)
        if abs(dz) > 0.1 and horizontal_distance < 0.1:
            return SegmentType.VERTICAL_UP if dz > 0 else SegmentType.VERTICAL_DOWN

        # Horizontal movement based on mission type
        elif horizontal_distance > 0.1:
            if 'square' in mission_name:
                return SegmentType.HORIZONTAL_STRAIGHT
            elif 'triangle' in mission_name:
                return SegmentType.HORIZONTAL_DIAGONAL
            else:
                # Default classification based on movement direction
                if abs(dx) < 0.05 or abs(dy) < 0.05:
                    return SegmentType.HORIZONTAL_STRAIGHT
                else:
                    return SegmentType.HORIZONTAL_DIAGONAL

        # Default case for small movements
        return SegmentType.HORIZONTAL_STRAIGHT

    def find_waypoint_arrival(self, states: List[Dict], waypoint: Tuple[float, float, float],
                              start_idx: int = 0) -> int:
        """Find the data index when waypoint is reached"""
        min_distance = float('inf')
        best_idx = len(states) - 1

        for i in range(start_idx, len(states)):
            state = states[i]
            position = state['components']['pos']
            velocity = state['components']['vel']
            distance = self.calculate_distance(position, waypoint)

            # Track minimum distance
            if distance < min_distance:
                min_distance = distance
                best_idx = i

            # Check if strict arrival condition is met
            if self.is_waypoint_reached(position, waypoint, velocity):
                return i

        # If not strictly reached, use relaxed condition (distance < 0.15m)
        if min_distance < 0.15:
            return best_idx

        return len(states) - 1

    def find_highest_point(self, states: List[Dict], start_idx: int = 0, end_idx: int = None) -> int:
        """Find the index of the highest point in the trajectory segment"""
        if end_idx is None:
            end_idx = len(states) - 1

        max_height = -float('inf')
        max_height_idx = start_idx

        for i in range(start_idx, min(end_idx + 1, len(states))):
            height = states[i]['components']['pos'][2]
            if height > max_height:
                max_height = height
                max_height_idx = i

        return max_height_idx

    def is_apex_waypoint(self, waypoint: Tuple[float, float, float],
                         waypoints: List[Tuple[float, float, float]]) -> bool:
        """Check if waypoint is the highest point in the mission"""
        waypoint_height = waypoint[2]
        max_height = max(wp[2] for wp in waypoints)
        return abs(waypoint_height - max_height) < 0.05

    def split_takeoff_into_segments(self, states: List[Dict], takeoff_start_idx: int, takeoff_end_idx: int) -> List[
        Tuple[int, int, SegmentType]]:
        """Split takeoff segment into takeoff + vertical_up for all trajectory types"""
        if takeoff_end_idx <= takeoff_start_idx:
            return [(takeoff_start_idx, takeoff_end_idx, SegmentType.TAKEOFF)]

        # Extract vertical position and velocity
        heights = []
        vertical_velocities = []

        for i in range(takeoff_start_idx, takeoff_end_idx + 1):
            heights.append(states[i]['components']['pos'][2])
            vertical_velocities.append(states[i]['components']['vel'][2])

        heights = np.array(heights)
        vertical_velocities = np.array(vertical_velocities)

        # Find the transition point from initial takeoff to sustained vertical climb
        transition_idx = takeoff_start_idx
        min_segment_length = 8  # Minimum points for each segment

        # Look for sustained vertical velocity phase
        for i in range(min_segment_length, len(vertical_velocities) - min_segment_length):
            # Check for sustained positive vertical velocity
            window_size = 5
            if i + window_size < len(vertical_velocities):
                velocity_window = vertical_velocities[i:i + window_size]

                # Criteria for vertical_up phase:
                # 1. Sustained positive vertical velocity
                # 2. Relatively stable velocity (not accelerating too much)
                if (np.all(velocity_window > 0.2) and
                        np.std(velocity_window) < 0.15 and
                        np.mean(velocity_window) > 0.3):
                    transition_idx = takeoff_start_idx + i
                    break

        # Ensure both segments have minimum length
        remaining_length = takeoff_end_idx - transition_idx
        if remaining_length < min_segment_length:
            transition_idx = takeoff_end_idx - min_segment_length

        if transition_idx - takeoff_start_idx < min_segment_length:
            transition_idx = takeoff_start_idx + min_segment_length

        # Create segment list
        segments = []

        # Always keep initial takeoff segment
        segments.append((takeoff_start_idx, transition_idx, SegmentType.TAKEOFF))

        # Add vertical_up segment if there's enough data
        if takeoff_end_idx > transition_idx:
            segments.append((transition_idx, takeoff_end_idx, SegmentType.VERTICAL_UP))

        return segments

    def segment_vertical_trajectory(self, data: Dict[str, Any], verbose: bool = True) -> List[TrajectorySegment]:
        """Vertical trajectory segmentation with apex correction and takeoff splitting"""
        states = data['data']['states']
        waypoints = data['mission']['waypoints']
        mission_name = data['mission']['name']

        segments = []
        segment_id = 0

        if verbose:
            print(f"Segmenting VERTICAL trajectory with takeoff splitting: {mission_name}")
            print(f"Waypoint sequence: {waypoints}")
            print(f"Number of waypoints: {len(waypoints)}")
            print(f"Number of state data points: {len(states)}")

        # Takeoff segmentation
        takeoff_end_idx = self.find_waypoint_arrival(states, waypoints[0])
        takeoff_segments = self.split_takeoff_into_segments(states, 0, takeoff_end_idx)

        for start_idx, end_idx, seg_type in takeoff_segments:
            if seg_type == SegmentType.TAKEOFF:
                end_waypoint = tuple(states[end_idx]['components']['pos'])
            else:  # VERTICAL_UP
                end_waypoint = waypoints[0]

            segments.append(self.create_segment(
                segment_id, states, start_idx, end_idx,
                seg_type, (0, 0, 0), end_waypoint
            ))

            if verbose:
                print(f"Created {seg_type.value} segment {segment_id} ({start_idx} -> {end_idx})")
            segment_id += 1

        # Process waypoint-to-waypoint segments
        start_idx = takeoff_end_idx if takeoff_end_idx > 0 else 0
        current_waypoint_idx = 1

        while current_waypoint_idx < len(waypoints):
            current_waypoint = waypoints[current_waypoint_idx]
            previous_waypoint = waypoints[current_waypoint_idx - 1]

            # Check if current waypoint is the apex
            is_apex = self.is_apex_waypoint(current_waypoint, waypoints)

            if is_apex:
                # For apex waypoint, find the true highest point in the entire remaining trajectory
                true_apex_idx = self.find_highest_point(states, start_idx, len(states) - 1)
                true_apex_pos = tuple(states[true_apex_idx]['components']['pos'])

                if verbose:
                    print(f"Apex waypoint detected at {current_waypoint}")
                    print(f"True apex found at index {true_apex_idx}, height: {true_apex_pos[2]:.3f}m")

                # Create vertical_up segment to true apex
                if true_apex_idx > start_idx:
                    segments.append(self.create_segment(
                        segment_id, states, start_idx, true_apex_idx,
                        SegmentType.VERTICAL_UP, previous_waypoint, true_apex_pos
                    ))
                    if verbose:
                        print(f"Created VERTICAL_UP segment {segment_id} ({start_idx} -> {true_apex_idx})")
                    segment_id += 1

                # Skip current apex waypoint and find next waypoint for descent
                next_waypoint_idx = current_waypoint_idx + 1
                if next_waypoint_idx < len(waypoints):
                    next_waypoint = waypoints[next_waypoint_idx]
                    next_waypoint_arrival_idx = self.find_waypoint_arrival(states, next_waypoint, true_apex_idx)

                    # Create vertical_down segment from true apex to next waypoint
                    if next_waypoint_arrival_idx > true_apex_idx:
                        segments.append(self.create_segment(
                            segment_id, states, true_apex_idx, next_waypoint_arrival_idx,
                            SegmentType.VERTICAL_DOWN, true_apex_pos, next_waypoint
                        ))
                        if verbose:
                            print(
                                f"Created VERTICAL_DOWN segment {segment_id} ({true_apex_idx} -> {next_waypoint_arrival_idx}) to next waypoint")
                        segment_id += 1
                        start_idx = next_waypoint_arrival_idx
                        current_waypoint_idx = next_waypoint_idx  # Skip the apex waypoint
                    else:
                        start_idx = true_apex_idx
                else:
                    # No next waypoint, this will be handled by final descent
                    start_idx = true_apex_idx

            else:
                # Regular waypoint processing
                end_idx = self.find_waypoint_arrival(states, current_waypoint, start_idx)

                if verbose:
                    print(
                        f"Regular segment from waypoint {current_waypoint_idx - 1} to {current_waypoint_idx}: index {start_idx} -> {end_idx}")

                if end_idx > start_idx:
                    height_diff = current_waypoint[2] - previous_waypoint[2]

                    if height_diff > 0.05:
                        segment_type = SegmentType.VERTICAL_UP
                    elif height_diff < -0.05:
                        segment_type = SegmentType.VERTICAL_DOWN
                    else:
                        segment_type = SegmentType.VERTICAL_DOWN

                    segments.append(self.create_segment(
                        segment_id, states, start_idx, end_idx,
                        segment_type, previous_waypoint, current_waypoint
                    ))

                    if verbose:
                        print(f"Created {segment_type.value} segment {segment_id}")

                    segment_id += 1
                    start_idx = end_idx

            current_waypoint_idx += 1

        # Final descent segment
        if start_idx < len(states) - 1:
            segments.append(self.create_segment(
                segment_id, states, start_idx, len(states) - 1,
                SegmentType.VERTICAL_DOWN, waypoints[-1], (0, 0, 0.2)
            ))
            if verbose:
                print(f"Final descent segment {segment_id}: VERTICAL_DOWN (index {start_idx} -> {len(states) - 1})")

        if verbose:
            print(f"Total segments created: {len(segments)}")
            print()

        return segments

    def segment_horizontal_trajectory(self, data: Dict[str, Any], verbose: bool = True) -> List[TrajectorySegment]:
        """Horizontal trajectory segmentation with takeoff splitting"""
        states = data['data']['states']
        waypoints = data['mission']['waypoints']
        mission_name = data['mission']['name']

        segments = []
        segment_id = 0

        if verbose:
            print(f"Segmenting HORIZONTAL trajectory with takeoff splitting: {mission_name}")
            print(f"Using takeoff segmentation")
            print(f"Number of waypoints: {len(waypoints)}")
            print(f"Number of state data points: {len(states)}")

        # Takeoff segmentation (same as vertical)
        takeoff_end_idx = self.find_waypoint_arrival(states, waypoints[0])
        takeoff_segments = self.split_takeoff_into_segments(states, 0, takeoff_end_idx)

        for start_idx, end_idx, seg_type in takeoff_segments:
            if seg_type == SegmentType.TAKEOFF:
                end_waypoint = tuple(states[end_idx]['components']['pos'])
            else:  # VERTICAL_UP
                end_waypoint = waypoints[0]

            segments.append(self.create_segment(
                segment_id, states, start_idx, end_idx,
                seg_type, (0, 0, 0), end_waypoint
            ))

            if verbose:
                print(f"Created {seg_type.value} segment {segment_id} ({start_idx} -> {end_idx})")
            segment_id += 1

        # Inter-waypoint flight segments
        start_idx = takeoff_end_idx if takeoff_end_idx > 0 else 0
        current_waypoint_idx = 1

        while current_waypoint_idx < len(waypoints):
            # Find arrival time at current waypoint
            end_idx = self.find_waypoint_arrival(states, waypoints[current_waypoint_idx], start_idx)

            if verbose:
                print(
                    f"Segment from waypoint {current_waypoint_idx - 1} to {current_waypoint_idx}: index {start_idx} -> {end_idx}")

            if end_idx > start_idx:
                # Determine segment type
                start_wp = waypoints[current_waypoint_idx - 1]
                end_wp = waypoints[current_waypoint_idx]
                segment_type = self.classify_segment_type(start_wp, end_wp, mission_name)

                segments.append(self.create_segment(
                    segment_id, states, start_idx, end_idx,
                    segment_type, start_wp, end_wp
                ))

                if verbose:
                    print(f"Created segment {segment_id}: {segment_type.value}")

                segment_id += 1
                start_idx = end_idx

            current_waypoint_idx += 1

        # Final segment: from last waypoint to end (always vertical_down)
        if start_idx < len(states) - 1:
            final_segment_type = SegmentType.VERTICAL_DOWN
            segments.append(self.create_segment(
                segment_id, states, start_idx, len(states) - 1,
                final_segment_type, waypoints[-1], (0, 0, 0.2)
            ))
            if verbose:
                print(
                    f"Final segment {segment_id}: {final_segment_type.value} (index {start_idx} -> {len(states) - 1})")

        if verbose:
            print(f"Total segments created: {len(segments)}")
            print()

        return segments

    def segment_trajectory(self, data: Dict[str, Any], verbose: bool = True) -> List[TrajectorySegment]:
        """Main segmentation function that routes to appropriate method"""
        mission_name = data['mission']['name']

        if 'vertical' in mission_name:
            return self.segment_vertical_trajectory(data, verbose)
        else:
            return self.segment_horizontal_trajectory(data, verbose)

    def create_segment(self, segment_id: int, states: List[Dict], start_idx: int, end_idx: int,
                       segment_type: SegmentType, start_wp: Tuple[float, float, float],
                       end_wp: Tuple[float, float, float]) -> TrajectorySegment:
        """Create trajectory segment object"""
        segment_states = states[start_idx:end_idx + 1]
        start_time = states[start_idx]['time']
        end_time = states[end_idx]['time']
        duration = end_time - start_time

        # Calculate distance and average speed
        distance = self.calculate_distance(
            states[start_idx]['components']['pos'],
            states[end_idx]['components']['pos']
        )
        avg_speed = distance / duration if duration > 0 else 0

        return TrajectorySegment(
            segment_id=segment_id,
            start_time=start_time,
            end_time=end_time,
            start_idx=start_idx,
            end_idx=end_idx,
            segment_type=segment_type,
            start_waypoint=start_wp,
            end_waypoint=end_wp,
            duration=duration,
            distance=distance,
            avg_speed=avg_speed,
            states=segment_states
        )

    def visualize_segmentation(self, data: Dict[str, Any], segments: List[TrajectorySegment],
                               save_path: str = None):
        """Visualize trajectory segmentation"""
        states = data['data']['states']
        waypoints = data['mission']['waypoints']
        mission_name = data['mission']['name']

        # Create figure with subplots
        if 'vertical' in mission_name:
            # For vertical trajectories: show height vs time
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Plot 1: Height vs Time with segments
            times = [s['time'] for s in states]
            heights = [s['components']['pos'][2] for s in states]
            times = [t - times[0] for t in times]  # Normalize time

            ax1.plot(times, heights, 'b-', alpha=0.6, linewidth=1, label='Complete trajectory')

            # Plot segments with different colors
            colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'purple', 'brown', 'pink']
            for i, segment in enumerate(segments):
                seg_times = [s['time'] - states[0]['time'] for s in segment.states]
                seg_heights = [s['components']['pos'][2] for s in segment.states]

                color = colors[i % len(colors)]
                ax1.plot(seg_times, seg_heights, color=color, linewidth=3, alpha=0.8,
                         label=f'Seg{i}: {segment.segment_type.value}')

            # Mark waypoint heights
            if waypoints:
                wp_heights = [wp[2] for wp in waypoints]
                max_time = max(times)
                wp_times = [i * max_time / (len(wp_heights) - 1) for i in range(len(wp_heights))]
                ax1.scatter(wp_times, wp_heights, c='red', s=100, marker='o', zorder=5, label='Target waypoints')

            # Mark actual highest point
            global_highest_idx = self.find_highest_point(states)
            highest_time = states[global_highest_idx]['time'] - states[0]['time']
            highest_height = states[global_highest_idx]['components']['pos'][2]
            ax1.scatter([highest_time], [highest_height], c='black', s=150, marker='*', zorder=6,
                        label='Actual highest point')

            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Height (m)')
            ax1.set_title(f'{mission_name} - Height vs Time (Segmentation)')
            ax1.grid(True)
            ax1.legend()

        else:
            # For horizontal trajectories: show X-Y plane
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Plot 1: X-Y trajectory with segments
            positions = np.array([s['components']['pos'] for s in states])
            ax1.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.6, linewidth=1, label='Complete trajectory')

            # Plot segments with different colors
            colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'purple', 'brown', 'pink']
            for i, segment in enumerate(segments):
                seg_positions = np.array([s['components']['pos'] for s in segment.states])

                color = colors[i % len(colors)]
                ax1.plot(seg_positions[:, 0], seg_positions[:, 1], color=color, linewidth=3, alpha=0.8,
                         label=f'Seg{i}: {segment.segment_type.value}')

            # Mark waypoints
            if waypoints:
                wp_array = np.array(waypoints)
                ax1.scatter(wp_array[:, 0], wp_array[:, 1], c='red', s=100, marker='o', zorder=5, label='Waypoints')

            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_title(f'{mission_name} - X-Y Trajectory (Segmentation)')
            ax1.grid(True)
            ax1.axis('equal')
            ax1.legend()

        # Plot 2: Speed vs Time (common for all trajectories)
        times = [s['time'] for s in states]
        speeds = [self.calculate_speed(s['components']['vel']) for s in states]
        times = [t - times[0] for t in times]  # Normalize time

        ax2.plot(times, speeds, 'g-', alpha=0.6, linewidth=1)

        # Mark segment boundaries
        for segment in segments:
            start_time = segment.start_time - states[0]['time']
            ax2.axvline(start_time, color='red', linestyle='--', alpha=0.7)
            ax2.text(start_time, max(speeds) * 0.9, segment.segment_type.value,
                     rotation=90, fontsize=8, ha='right')

        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Speed (m/s)')
        ax2.set_title('Speed vs Time with Segment Boundaries')
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization: {save_path}")

        plt.show()

    def print_segment_summary(self, segments: List[TrajectorySegment]):
        """Print detailed segment analysis"""
        print("=" * 80)
        print("TRAJECTORY SEGMENTATION SUMMARY")
        print("=" * 80)

        total_duration = sum(s.duration for s in segments)
        total_distance = sum(s.distance for s in segments)

        print(f"Total segments: {len(segments)}")
        print(f"Total duration: {total_duration:.2f}s")
        print(f"Total distance: {total_distance:.2f}m")
        print(f"Average speed: {total_distance / total_duration:.2f}m/s")
        print()

        print(
            f"{'ID':<3} {'Type':<20} {'Duration(s)':<11} {'Distance(m)':<12} {'Speed(m/s)':<11} {'Start(s)':<9} {'End(s)':<7}")
        print("-" * 80)

        for segment in segments:
            print(f"{segment.segment_id:<3} {segment.segment_type.value:<20} "
                  f"{segment.duration:<11.2f} {segment.distance:<12.2f} {segment.avg_speed:<11.2f} "
                  f"{segment.start_time:<9.2f} {segment.end_time:<7.2f}")
        print()

    def export_segments_for_training(self, segments: List[TrajectorySegment], filename: str,
                                     output_dir: str = "segmented_training_data"):
        """Export segmented data for training"""
        os.makedirs(output_dir, exist_ok=True)

        # Prepare training data
        training_data = {
            "file_info": {
                "original_filename": filename,
                "total_segments": len(segments),
                "segment_types": [seg.segment_type.value for seg in segments]
            },
            "segments": []
        }

        for segment in segments:
            segment_data = {
                "segment_info": segment.to_dict(),
                "trajectory_points": []
            }

            # Extract trajectory points with relative time
            start_time = segment.start_time
            for state in segment.states:
                point = {
                    "time": state['time'] - start_time,  # Relative time within segment
                    "position": state['components']['pos'],
                    "velocity": state['components']['vel'],
                    "attitude": state['components']['att'],
                    "acceleration": state['components']['acc']
                }
                segment_data["trajectory_points"].append(point)

            training_data["segments"].append(segment_data)

        # Save training data
        output_path = os.path.join(output_dir, f"{filename.replace('.json', '_segmented.json')}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)

        print(f"Exported segmented data: {output_path}")
        return output_path

    def process_single_subcategory(self, selected_dir: str = "selected_trajectories", subcategory: str = None):
        """Process a single subcategory"""
        if not os.path.exists(selected_dir):
            print(f"Selected trajectories directory not found: {selected_dir}")
            return

        # Find all subdirectories
        subdirs = [d for d in os.listdir(selected_dir)
                   if os.path.isdir(os.path.join(selected_dir, d))]

        if subcategory:
            # Filter to specific subcategory
            target_subdir = subcategory
            if target_subdir not in subdirs:
                print(f"Subcategory '{subcategory}' not found.")
                print(f"Available subcategories: {subdirs}")
                return
            subdirs = [target_subdir]

        print(f"Processing subcategory: {subdirs[0]}")
        print("=" * 60)

        all_results = []

        for subdir in subdirs:
            subdir_path = os.path.join(selected_dir, subdir)
            json_files = glob.glob(os.path.join(subdir_path, "*.json"))

            # Skip metadata files
            json_files = [f for f in json_files if not f.endswith("_metadata.json")]

            if not json_files:
                print(f"No JSON files found in {subdir}")
                continue

            print(f"\nProcessing {subdir}: {len(json_files)} files")
            print("-" * 40)

            for json_file in sorted(json_files):
                filename = os.path.basename(json_file)
                print(f"\n>> {filename}")

                # Load and segment trajectory
                data = self.load_json_data(json_file)
                if data is None:
                    continue

                segments = self.segment_trajectory(data, verbose=True)

                # Print summary
                self.print_segment_summary(segments)

                # Create visualization
                viz_path = json_file.replace('.json', '_segmentation.png')
                self.visualize_segmentation(data, segments, viz_path)

                # Export for training
                training_path = self.export_segments_for_training(segments, filename)

                # Store results
                result = {
                    'subcategory': subdir,
                    'filename': filename,
                    'segments': [seg.to_dict() for seg in segments],
                    'training_data_path': training_path
                }
                all_results.append(result)

            print(f"\nCompleted {subdir}: processed {len(json_files)} trajectories")

        # Save subcategory summary
        summary_path = os.path.join(selected_dir, f"segmentation_summary_{subcategory}.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    segmenter = DroneTrajectorySegmenter()

    # Show available subcategories
    selected_dir = "selected_trajectories"
    if os.path.exists(selected_dir):
        subdirs = [d for d in os.listdir(selected_dir)
                   if os.path.isdir(os.path.join(selected_dir, d))]

        print("Available subcategories:")
        for i, subdir in enumerate(sorted(subdirs), 1):
            print(f"{i}. {subdir}")

        choice = input(f"\nEnter choice (1-{len(subdirs)}): ").strip()

        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(subdirs):
                selected_subcategory = sorted(subdirs)[choice_idx]
                print(f"\nProcessing: {selected_subcategory}")
                segmenter.process_single_subcategory(selected_dir, selected_subcategory)
            else:
                print("Invalid choice.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    else:
        print(f"Directory {selected_dir} not found.")

    print("\nSegmentation analysis completed!")