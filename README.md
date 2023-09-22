# Rolling Shutter Simulation Attack

This repository provides the materials and results for a rolling shutter attack based on a real experiment designed similarly to the simulation.

## Repository Contents:

- `RollingShutterSimAttack.py`: The Python simulation code that models the rolling shutter attack.
- `laser_pulse.mat`: Contains the image of a single pulse, which is essential for running the simulation.
- `PulseWidthResults`: A directory containing experimental results categorized by the pulse width in microseconds.
  - Subfolders (named by pulse width: `1`, `25`, `50`, `70`, `100`) each contain:
    - An `.mp4` video (e.g., `test_50microSec.mp4`) displaying the experimental outcome for that specific pulse width.
    - An accompanying `.jpg` image (e.g., `test_50microSec.jpg`) that provides an analysis of the corresponding video.

## Prerequisites

Ensure you have the following libraries installed to run the simulation code:

```sh
torch
Pillow
scipy
numpy
torchvision
```

To install the above libraries, you can use pip:

```sh
pip install torch Pillow scipy numpy torchvision
```

## Running the Code

To run the simulation, navigate to the repository directory in your terminal and run the following command:

```sh
python RollingShutterSimAttack.py
```

## Further Information

Please refer to the individual subfolders for the specific results and consult the `video_and_image_descriptions.md` within `PulseWidthResults` for detailed descriptions of each video and its associated analysis.

