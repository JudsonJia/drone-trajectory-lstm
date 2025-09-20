import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import shutil
from typing import List, Dict, Any, Optional
import matplotlib

matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class TrajectorySelector:
    def __init__(self):
        self.trajectories = []
        self.selected_indices = []
        self.current_category = None
        self.current_size = None

    def load_json_data(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load JSON data file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def calculate_speed(self, velocity: List[float]) -> float:
        """Calculate speed magnitude"""
        return np.sqrt(sum(v ** 2 for v in velocity))

    def smooth_velocity_data(self, states: List[Dict], window_size: int = 3) -> List[Dict]:
        """Smooth velocity data to reduce noise"""
        if len(states) < window_size:
            return states

        smoothed_states = []
        half_window = window_size // 2

        for i in range(len(states)):
            new_state = states[i].copy()
            new_state['components'] = states[i]['components'].copy()

            if half_window <= i < len(states) - half_window:
                vel_sum = [0.0, 0.0, 0.0]
                count = 0

                for j in range(i - half_window, i + half_window + 1):
                    if 0 <= j < len(states):
                        for k in range(3):
                            vel_sum[k] += states[j]['components']['vel'][k]
                        count += 1

                if count > 0:
                    new_state['components']['vel'] = [v / count for v in vel_sum]

            smoothed_states.append(new_state)

        return smoothed_states

    def normalize_time_axis(self, times: List[float]) -> List[float]:
        """Normalize time to start from 0"""
        if not times:
            return times
        start_time = times[0]
        return [t - start_time for t in times]

    def process_trajectory_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Process a single trajectory file"""
        data = self.load_json_data(file_path)
        if data is None:
            return None

        # Check if trajectory is valid (5-25 seconds)
        states = data['data']['states']
        if not states:
            return None

        total_duration = states[-1]['time'] - states[0]['time']
        if not (5.0 <= total_duration <= 25.0):
            return None

        # Smooth velocity data
        data['data']['states'] = self.smooth_velocity_data(states)

        # Extract trajectory data
        states = data['data']['states']
        positions = np.array([s['components']['pos'] for s in states])
        times = [s['time'] for s in states]
        heights = [s['components']['pos'][2] for s in states]
        speeds = [self.calculate_speed(s['components']['vel']) for s in states]

        # Normalize time
        times = self.normalize_time_axis(times)

        # Clip speeds to reasonable range
        speeds = np.clip(speeds, 0, 1.0)

        return {
            'file_path': file_path,
            'filename': os.path.basename(file_path),
            'mission_name': data['mission']['name'],
            'positions': positions,
            'times': times,
            'heights': heights,
            'speeds': speeds,
            'waypoints': data['mission']['waypoints'],
            'duration': total_duration
        }

    def find_files_by_category(self, base_path: str) -> Dict[str, Dict[str, List[str]]]:
        """Find all JSON files organized by category and size"""
        file_structure = {}
        categories = ['vertical', 'square', 'triangle']
        sizes = ['small', 'med', 'large']

        for category in categories:
            file_structure[category] = {}
            for size in sizes:
                pattern = os.path.join(base_path, category, size, f"{category}_{size}_*.json")
                files = glob.glob(pattern)
                file_structure[category][size] = sorted(files)

        return file_structure

    def load_category_trajectories(self, base_path: str, category: str, size: str) -> List[Dict]:
        """Load all trajectories for a specific category and size"""
        pattern = os.path.join(base_path, category, size, f"{category}_{size}_*.json")
        files = glob.glob(pattern)

        trajectories = []
        print(f"Loading {category}_{size} trajectories...")

        for i, file_path in enumerate(files):
            if i % 10 == 0:
                print(f"  Processing file {i + 1}/{len(files)}")
            trajectory = self.process_trajectory_file(file_path)
            if trajectory is not None:
                trajectories.append(trajectory)

        print(f"Loaded {len(trajectories)} valid trajectories")
        return trajectories

    def create_preview_grid(self, trajectories: List[Dict], category: str, size: str,
                            trajectories_per_page: int = 20):
        """Create a grid of trajectory previews"""
        num_pages = (len(trajectories) + trajectories_per_page - 1) // trajectories_per_page

        # Get current working directory
        current_dir = os.getcwd()
        print(f"Creating previews in directory: {current_dir}")

        for page in range(num_pages):
            start_idx = page * trajectories_per_page
            end_idx = min(start_idx + trajectories_per_page, len(trajectories))
            page_trajectories = trajectories[start_idx:end_idx]

            print(f"Creating page {page + 1}/{num_pages} with trajectories {start_idx}-{end_idx - 1}...")

            # Create subplot grid
            cols = 5
            rows = (len(page_trajectories) + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
            if rows == 1:
                axes = axes.reshape(1, -1)

            # Set title based on category
            view_type = "Height vs Time" if category == "vertical" else "X-Y Plane"
            fig.suptitle(
                f'{category.title()}_{size} - Page {page + 1}/{num_pages} ({view_type}) - Trajectories {start_idx}-{end_idx - 1}',
                fontsize=16)

            for i, traj in enumerate(page_trajectories):
                row = i // cols
                col = i % cols
                ax = axes[row, col] if rows > 1 else axes[col]

                if category == "vertical":
                    # For vertical trajectories, show height vs time
                    times = traj['times']
                    heights = traj['heights']
                    ax.plot(times, heights, 'b-', linewidth=2, alpha=0.8)

                    # Mark waypoints on height plot
                    if traj['waypoints']:
                        wp_heights = [wp[2] for wp in traj['waypoints']]
                        # Find approximate times for waypoints (simplified)
                        max_time = max(times) if times else 1
                        wp_times = [i * max_time / (len(wp_heights) - 1) for i in range(len(wp_heights))]
                        ax.scatter(wp_times, wp_heights, c='red', s=50, marker='o', zorder=5)

                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Height (m)')
                    ax.set_xlim(0, 20)
                    if heights:
                        ax.set_ylim(0, max(heights) * 1.1)

                else:
                    # For square and triangle trajectories, show X-Y plane
                    positions = traj['positions']
                    ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, alpha=0.8)

                    # Plot waypoints
                    if traj['waypoints']:
                        wp_array = np.array(traj['waypoints'])
                        ax.scatter(wp_array[:, 0], wp_array[:, 1], c='red', s=50, marker='o', zorder=5)

                    ax.set_xlabel('X (m)')
                    ax.set_ylabel('Y (m)')
                    ax.axis('equal')

                ax.set_title(f'#{start_idx + i}: {traj["filename"][:15]}...', fontsize=10)
                ax.grid(True, alpha=0.3)

            # Hide empty subplots
            for i in range(len(page_trajectories), rows * cols):
                row = i // cols
                col = i % cols
                ax = axes[row, col] if rows > 1 else axes[col]
                ax.set_visible(False)

            plt.tight_layout()

            # Save preview with full path
            preview_filename = f'trajectory_preview_{category}_{size}_page{page + 1}.png'
            preview_path = os.path.join(current_dir, preview_filename)
            plt.savefig(preview_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Saved preview: {preview_path}")

        print(f"\nPreview generation completed!")
        print(f"Generated {num_pages} preview images showing:")
        if category == "vertical":
            print("  - Height vs Time plots (showing vertical motion patterns)")
        else:
            print("  - X-Y plane plots (showing horizontal motion patterns)")
        print(f"Look for files named: trajectory_preview_{category}_{size}_page*.png")
        print(f"Full path: {current_dir}")

        # Try to open file explorer (Windows only)
        try:
            if os.name == 'nt':  # Windows
                os.startfile(current_dir)
                print("Opened file explorer to preview directory.")
        except:
            pass

    def interactive_selection_by_index(self, trajectories: List[Dict], category: str, size: str):
        """Interactive selection by trajectory index"""
        self.trajectories = trajectories
        self.current_category = category
        self.current_size = size
        self.selected_indices = []

        print(f"\n=== Trajectory Selection for {category}_{size} ===")
        print(f"Total trajectories: {len(trajectories)}")
        print("\nCommands:")
        print("  <number>     - Select trajectory by index (e.g., 5)")
        print("  <start-end>  - Select range (e.g., 10-15)")
        print("  list         - Show selected trajectories")
        print("  preview      - Create preview images")
        print("  save         - Save selected trajectories")
        print("  info <num>   - Show detailed info for trajectory")
        print("  quit         - Exit selection")

        while True:
            command = input(f"\nSelected: {len(self.selected_indices)} trajectories. Enter command: ").strip()

            if command.lower() == 'quit':
                break
            elif command.lower() == 'list':
                self.show_selected_list()
            elif command.lower() == 'preview':
                self.create_preview_grid(trajectories, category, size)
            elif command.lower() == 'save':
                self.save_selected_trajectories()
            elif command.lower().startswith('info '):
                try:
                    idx = int(command.split()[1])
                    self.show_trajectory_info(idx)
                except (ValueError, IndexError):
                    print("Usage: info <trajectory_number>")
            elif '-' in command:
                # Range selection
                try:
                    start, end = map(int, command.split('-'))
                    self.select_range(start, end)
                except ValueError:
                    print("Invalid range format. Use: start-end (e.g., 10-15)")
            else:
                # Single number selection
                try:
                    idx = int(command)
                    self.toggle_selection(idx)
                except ValueError:
                    print("Invalid command. Enter a number, range, or command.")

    def toggle_selection(self, idx: int):
        """Toggle selection of a single trajectory"""
        if 0 <= idx < len(self.trajectories):
            if idx in self.selected_indices:
                self.selected_indices.remove(idx)
                print(f"Deselected #{idx}: {self.trajectories[idx]['filename']}")
            else:
                self.selected_indices.append(idx)
                print(f"Selected #{idx}: {self.trajectories[idx]['filename']}")
        else:
            print(f"Invalid index {idx}. Valid range: 0-{len(self.trajectories) - 1}")

    def select_range(self, start: int, end: int):
        """Select a range of trajectories"""
        if start > end:
            start, end = end, start

        count = 0
        for idx in range(start, end + 1):
            if 0 <= idx < len(self.trajectories):
                if idx not in self.selected_indices:
                    self.selected_indices.append(idx)
                    count += 1

        print(f"Selected {count} trajectories in range {start}-{end}")

    def show_selected_list(self):
        """Show list of selected trajectories"""
        if not self.selected_indices:
            print("No trajectories selected.")
            return

        print(f"\nSelected trajectories ({len(self.selected_indices)}):")
        for idx in sorted(self.selected_indices):
            traj = self.trajectories[idx]
            print(f"  #{idx}: {traj['filename']} (duration: {traj['duration']:.1f}s)")

    def show_trajectory_info(self, idx: int):
        """Show detailed information about a trajectory"""
        if 0 <= idx < len(self.trajectories):
            traj = self.trajectories[idx]
            print(f"\nTrajectory #{idx} Info:")
            print(f"  Filename: {traj['filename']}")
            print(f"  Duration: {traj['duration']:.2f}s")
            print(f"  Points: {len(traj['positions'])}")
            print(f"  Max height: {max(traj['heights']):.2f}m")
            print(f"  Max speed: {max(traj['speeds']):.2f}m/s")
            print(f"  Waypoints: {len(traj['waypoints'])}")
        else:
            print(f"Invalid index {idx}. Valid range: 0-{len(self.trajectories) - 1}")

    def save_selected_trajectories(self, output_dir: str = "selected_trajectories"):
        """Save the selected trajectories to a new directory"""
        if not self.selected_indices:
            print("No trajectories selected!")
            return

        os.makedirs(output_dir, exist_ok=True)

        # Create subdirectory for this category and size
        category_dir = os.path.join(output_dir, f"{self.current_category}_{self.current_size}")
        os.makedirs(category_dir, exist_ok=True)

        print(f"Saving {len(self.selected_indices)} selected trajectories to {category_dir}...")

        # Copy selected files
        saved_files = []
        for idx in self.selected_indices:
            traj = self.trajectories[idx]
            file_path = traj['file_path']
            filename = os.path.basename(file_path)
            dest_path = os.path.join(category_dir, filename)
            shutil.copy2(file_path, dest_path)
            saved_files.append(filename)
            print(f"  Copied: {filename}")

        # Save selection metadata
        metadata = {
            'category': self.current_category,
            'size': self.current_size,
            'selected_count': len(self.selected_indices),
            'total_available': len(self.trajectories),
            'selected_indices': sorted(self.selected_indices),
            'selected_files': saved_files
        }

        metadata_path = os.path.join(category_dir, 'selection_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"\nSelection completed! Files saved to: {category_dir}")
        print(f"Metadata saved to: {metadata_path}")

    def run_selection_workflow(self, base_path: str):
        """Run the complete selection workflow"""
        print("=== Trajectory Selection Tool ===")

        file_structure = self.find_files_by_category(base_path)

        # Show available options
        print("Available categories and sizes:")
        for category, sizes in file_structure.items():
            print(f"\n{category.upper()}:")
            for size, files in sizes.items():
                print(f"  {size}: {len(files)} files")

        # Get user selection
        print("\nEnter the category and size you want to select from:")
        category = input("Category (vertical/square/triangle): ").strip().lower()
        size = input("Size (small/med/large): ").strip().lower()

        if category not in file_structure or size not in file_structure[category]:
            print("Invalid category or size!")
            return

        if not file_structure[category][size]:
            print(f"No files found for {category}_{size}")
            return

        # Load trajectories
        trajectories = self.load_category_trajectories(base_path, category, size)
        if not trajectories:
            print("No valid trajectories found!")
            return

        # Start interactive selection
        self.interactive_selection_by_index(trajectories, category, size)


if __name__ == "__main__":
    selector = TrajectorySelector()

    # base path
    base_path = r"C:\Users\51539\Desktop\Drone_Trajectory_LSTM\Flight_Data"

    # Start selection workflow
    selector.run_selection_workflow(base_path)