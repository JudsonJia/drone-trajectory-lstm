import json
import os
import glob
from collections import defaultdict, Counter
import pandas as pd


class SimpleDatasetStats:
    def __init__(self):
        self.stats = {
            'total_files': 0,
            'total_segments': 0,
            'trajectory_types': defaultdict(int),
            'segment_type_counts': defaultdict(int),
            'files_by_trajectory': defaultdict(list),
            'segments_per_trajectory': defaultdict(list)
        }

    def analyze_dataset(self, data_dir: str = "segmented_training_data"):
        """Analyze all segmented JSON files"""
        if not os.path.exists(data_dir):
            print(f"Data directory does not exist: {data_dir}")
            return

        json_files = glob.glob(os.path.join(data_dir, "*_segmented.json"))

        if not json_files:
            print(f"No segmented data files found in {data_dir}")
            return

        print(f"Starting analysis of {len(json_files)} files...")
        print("=" * 50)

        file_details = []

        for json_file in sorted(json_files):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                file_info = data.get('file_info', {})
                filename = file_info.get('original_filename', os.path.basename(json_file))
                total_segments = file_info.get('total_segments', 0)
                segment_types = file_info.get('segment_types', [])

                # Extract trajectory type
                trajectory_type = self.extract_trajectory_type(filename)

                # Update statistics
                self.stats['total_files'] += 1
                self.stats['total_segments'] += total_segments
                self.stats['trajectory_types'][trajectory_type] += 1
                self.stats['files_by_trajectory'][trajectory_type].append(filename)
                self.stats['segments_per_trajectory'][trajectory_type].append(total_segments)

                # Count segment types
                for seg_type in segment_types:
                    self.stats['segment_type_counts'][seg_type] += 1

                # Store file details
                file_details.append({
                    'filename': filename,
                    'trajectory_type': trajectory_type,
                    'total_segments': total_segments,
                    'segment_types': segment_types,
                    'segment_breakdown': dict(Counter(segment_types))
                })

                print(f"{filename}: {trajectory_type}, {total_segments} segments")

            except Exception as e:
                print(f"Error processing file {json_file}: {e}")

        self.print_summary()
        self.save_results(data_dir, file_details)

    def extract_trajectory_type(self, filename: str) -> str:
        """Extract trajectory type from filename"""
        filename_lower = filename.lower()

        # Determine base type
        if 'vertical' in filename_lower:
            base_type = 'vertical'
        elif 'square' in filename_lower:
            base_type = 'square'
        elif 'triangle' in filename_lower:
            base_type = 'triangle'
        else:
            return 'unknown'

        # Determine size
        if 'small' in filename_lower:
            size = 'small'
        elif 'medium' in filename_lower or 'med' in filename_lower:
            size = 'medium'
        elif 'large' in filename_lower:
            size = 'large'
        else:
            size = 'unknown'

        return f"{base_type}_{size}"

    def print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "=" * 50)
        print("Dataset Statistics Summary")
        print("=" * 50)

        print(f"Total files: {self.stats['total_files']}")
        print(f"Total segments: {self.stats['total_segments']}")
        print(f"Average segments per file: {self.stats['total_segments'] / self.stats['total_files']:.2f}")

        print(f"\nTrajectory type distribution:")
        for traj_type, count in sorted(self.stats['trajectory_types'].items()):
            avg_segments = sum(self.stats['segments_per_trajectory'][traj_type]) / count
            print(f"  {traj_type}: {count} files, avg {avg_segments:.1f} segments/file")

        print(f"\nSegment type statistics:")
        total_segments = sum(self.stats['segment_type_counts'].values())
        for seg_type, count in sorted(self.stats['segment_type_counts'].items()):
            percentage = (count / total_segments) * 100
            print(f"  {seg_type}: {count} segments ({percentage:.1f}%)")

        print(f"\nDetailed info by trajectory type:")
        for traj_type in sorted(self.stats['trajectory_types'].keys()):
            files = self.stats['files_by_trajectory'][traj_type]
            segments_list = self.stats['segments_per_trajectory'][traj_type]

            print(f"\n{traj_type}:")
            print(f"  File count: {len(files)}")
            print(f"  Total segments: {sum(segments_list)}")
            print(f"  Segment range: {min(segments_list)}-{max(segments_list)}")
            print(f"  File list: {files[:3]}{'...' if len(files) > 3 else ''}")

    def save_results(self, data_dir: str, file_details: list):
        """Save results to files"""

        # Save summary statistics
        summary_file = os.path.join(data_dir, "dataset_summary.json")
        summary_data = {
            'overview': {
                'total_files': self.stats['total_files'],
                'total_segments': self.stats['total_segments'],
                'avg_segments_per_file': self.stats['total_segments'] / self.stats['total_files']
            },
            'trajectory_type_distribution': dict(self.stats['trajectory_types']),
            'segment_type_distribution': dict(self.stats['segment_type_counts']),
            'detailed_file_info': file_details
        }

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        print(f"\nStatistics results saved to: {summary_file}")

        # Save as CSV for easy viewing
        csv_file = os.path.join(data_dir, "file_details.csv")

        # Flatten segment breakdown for CSV
        csv_data = []
        for file_detail in file_details:
            row = {
                'filename': file_detail['filename'],
                'trajectory_type': file_detail['trajectory_type'],
                'total_segments': file_detail['total_segments']
            }

            # Add segment type counts as separate columns
            segment_breakdown = file_detail['segment_breakdown']
            for seg_type in ['takeoff', 'vertical_up', 'vertical_down', 'horizontal_straight', 'horizontal_diagonal']:
                row[f'{seg_type}_count'] = segment_breakdown.get(seg_type, 0)

            csv_data.append(row)

        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"CSV details saved to: {csv_file}")


def main():
    analyzer = SimpleDatasetStats()

    data_directory = "segmented_training_data"

    if len(os.sys.argv) > 1:
        data_directory = os.sys.argv[1]

    analyzer.analyze_dataset(data_directory)

    print(f"\nAnalysis completed!")

if __name__ == "__main__":
    main()