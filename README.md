# Football Analysis Project

## Introduction

The goal of this project is to detect and track players, referees, and footballs in a video using YOLO, one of the best AI object detection models available. We will also train the model to improve its performance. Additionally, we will assign players to teams based on the colors of their t-shirts using Kmeans for pixel segmentation and clustering. With this information, we can measure a team's ball acquisition percentage in a match. We will also use optical flow to measure camera movement between frames, enabling us to accurately measure a player's movement. Furthermore, we will implement perspective transformation to represent the scene's depth and perspective, allowing us to measure a player's movement in meters rather than pixels. Finally, we will calculate a player's speed and the distance covered. This project covers various concepts and addresses real-world problems, making it suitable for both beginners and experienced machine learning engineers.

![Screenshot](output_videos/screenshot.png)

## Modules Used

The following modules are used in this project:

- YOLO: AI object detection model.
- Kmeans: Pixel segmentation and clustering to detect t-shirt color
- Optical Flow: Measure camera movement
- Perspective Transformation: Represent scene depth and perspective  
- Speed and distance calculation per player

## Trained Models

- [Trained Yolo v5](https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view?usp=sharing)

## Sample video

- [Sample input video](https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view?usp=sharing)

## Requirements

To run this project, you need to have the following requirements installed:

- Python 3.x
- ultralytics
- supervision
- OpenCV
- NumPy
- Matplotlib
- Pandas



To run this project, you need to have the following requirements installed:

- Python 3.x
- ultralytics
- supervision
- OpenCV
- NumPy
- Matplotlib
- Pandas

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/hungchann/football_analysis.git
   cd football_analysis
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the trained model:
   - Download the [Trained Yolo v5 model](https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view?usp=sharing)
   - Place it in the `models` directory

5. Download a sample video:
   - Download the [Sample input video](https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view?usp=sharing)
   - Place it in the `input_videos` directory

## Usage

Run the main analysis script:

   ```bash
   python3 main.py
   ```

The analyzed video will be saved in the `output_videos` directory.

### Customizing Colors

You can customize team colors and box colors by modifying the following lines in `main.py`:

```python
# Set custom team colors (RGB format)
team_assigner.set_team_colors([255, 0, 0], [0, 0, 255])  # Red for team 1, Blue for team 2

# Set custom box colors for player numbers (RGB format)
team_assigner.set_box_colors([0, 255, 0], [255, 255, 0])  # Green for team 1 boxes, Yellow for team 2 boxes
```

