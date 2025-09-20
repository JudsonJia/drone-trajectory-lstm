# Drone Trajectory LSTM

A deep learning framework for autonomous drone trajectory prediction and generation using LSTM neural networks. This system learns complex flight patterns from real drone data and generates accurate trajectory predictions for various flight maneuvers.

## Overview

This project implements a two-stage neural network architecture:
1. **Global Duration Predictor**: Estimates flight duration and temporal parameters for trajectory segments
2. **Segment-Specific Trajectory Generators**: Specialized LSTM networks for different flight maneuvers (takeoff, vertical, horizontal movements)

The system achieves high accuracy in predicting drone trajectories with Average Displacement Error (ADE) typically under 0.12 meters for well-trained segments.

## Script Overview

### Core Training and Evaluation Scripts

- **`trajectory_trainer.py`** - Main training script for both global duration predictor and segment-specific LSTM generators. Features multi-stage training, physics-informed loss functions, and enhanced horizontal trajectory optimization.

- **`ADE_calculation.py`** - Comprehensive evaluation system that calculates Average Displacement Error metrics across all trajectory segments, generates performance statistics by segment type, and creates comparison visualizations.

- **`trajectory_visualizer.py`** - Advanced visualization tool for trajectory comparison between predicted and ground truth paths. Automatically detects mission types and provides appropriate 3D plots, height analysis, and X-Y plane projections.

### Data Processing and Analysis

- **`segmenter.py`** - Automated trajectory segmentation tool that converts raw flight data into labeled training segments. Analyzes flight patterns, identifies segment boundaries (takeoff, vertical, horizontal movements), and generates structured training data with metadata.

- **`count_segments.py`** - Dataset analysis utility that provides detailed statistics on trajectory files, segment type distributions, and data quality metrics. Essential for understanding your dataset composition before training.

- **`trajectory_selection.py`** - Interactive trajectory curation tool for selecting high-quality training data. Features preview generation, manual selection interface, and metadata management for building curated datasets.

## Features

- **Segmented Learning**: Separate models for different flight patterns (takeoff, vertical, horizontal maneuvers)
- **Real-time Prediction**: Fast trajectory generation suitable for real-time control applications  
- **Physics-Informed Training**: Incorporates velocity-position consistency and boundary constraints
- **Comprehensive Evaluation**: ADE-based performance metrics and detailed trajectory visualization
- **Data Pipeline**: Complete preprocessing pipeline from raw flight data to training-ready segments

## Architecture

### Model Components

1. **GlobalDurationPredictor**: 
   - Input: Start position, end position, segment type, direction vector
   - Output: Duration, average speed, trajectory sampling points
   - Architecture: Feed-forward network with segment embeddings

2. **SegmentTrajectoryLSTM**:
   - Input: Start/end positions, velocities, segment context
   - Output: Full 6D trajectory (position + velocity)
   - Architecture: LSTM with time encoding and boundary enforcement

### Segment Types

- `takeoff`: Initial launch sequences
- `vertical_up`: Vertical ascent maneuvers  
- `vertical_down`: Vertical descent maneuvers
- `horizontal_straight`: Linear horizontal flight
- `horizontal_diagonal`: Diagonal and curved horizontal paths

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/drone-trajectory-lstm.git
cd drone-trajectory-lstm

# Install dependencies
pip install torch torchvision numpy matplotlib pandas scikit-learn

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

### Model Training

```bash
# Train global duration predictor
python trajectory_trainer.py
# Select option 1: Train Global Duration Predictor

# Train all segment-specific generators
python trajectory_trainer.py  
# Select option 3: Train all Trajectory Generators

# Or train specific segments
python trajectory_trainer.py
# Select option 2, then choose specific segment type
```

### Evaluation and Visualization

```bash
# Calculate ADE metrics by segment type
python ADE_calculation.py

# Visualize trajectory comparisons
python trajectory_visualizer.py
```

## Training Configuration

### Global Duration Predictor
- **Epochs**: 100 (with early stopping)
- **Batch Size**: 32
- **Learning Rate**: 0.001 with ReduceLROnPlateau
- **Architecture**: 128 hidden units, 3-layer encoder
- **Loss**: Combined MSE for duration, speed, and trajectory points

### Trajectory Generators
- **Epochs**: 150 (with early stopping)  
- **Batch Size**: 8
- **Learning Rate**: 0.0005 with ReduceLROnPlateau
- **Architecture**: 2-layer LSTM, 128 hidden units
- **Loss**: Position + velocity + physics + boundary + acceleration terms

### Enhanced Horizontal Training
The system includes special handling for horizontal trajectories with intermediate point correction:

```python
# Horizontal trajectory optimization
if segment_type in ['horizontal_straight', 'horizontal_diagonal']:
    # Enhanced boundary loss weight (15.0 vs 10.0)
    # Intermediate waypoint correction during training
    # Improved trajectory smoothness constraints
```

## Data Format

### Input Data Structure
```json
{
  "segments": [
    {
      "segment_info": {
        "segment_type": "horizontal_straight",
        "duration": 2.5,
        "distance": 0.85,
        "avg_speed": 0.34
      },
      "trajectory_points": [
        {
          "time": 0.125,
          "position": [0.1, 0.2, 0.4],
          "velocity": [0.05, 0.0, 0.0]
        }
      ]
    }
  ]
}
```

### Model Output
```python
# Single segment prediction
prediction = {
    'segment_type': 'horizontal_straight',
    'duration': 2.48,
    'num_points': 248,
    'positions': np.array([[x, y, z], ...]),  # Shape: (N, 3)
    'velocities': np.array([[vx, vy, vz], ...]),  # Shape: (N, 3)
    'times': np.array([0.0, 0.01, 0.02, ...])  # Shape: (N,)
}
```

## Performance Metrics

### Average Displacement Error (ADE)
The system evaluates trajectory quality using ADE metrics:

```bash
# Example ADE results by segment type
takeoff: 0.0252 m
vertical_up: 0.0587 m  
vertical_down: 0.0919 m
horizontal_straight: 0.0600 m
horizontal_diagonal: 0.01137 m
```

## Model Files Structure

```
models/
├── global_duration_predictor.pth          # Duration prediction model
└── segment_generators/
    ├── takeoff_generator.pth              # Takeoff trajectory generator
    ├── vertical_up_generator.pth          # Vertical ascent generator
    ├── vertical_down_generator.pth        # Vertical descent generator
    ├── horizontal_straight_generator.pth  # Straight horizontal generator
    └── horizontal_diagonal_generator.pth  # Diagonal horizontal generator
```

## Related Work

- [Crazyflie Data Collection](https://github.com/JudsonJia/crazyfly-data-collection) - Real drone data collection system
