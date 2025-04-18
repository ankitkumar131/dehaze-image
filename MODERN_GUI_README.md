# Advanced Image Dehazing Tool

A modern, feature-rich GUI application for image dehazing with interactive controls and a sleek dark theme.

## Features

- Modern dark theme UI with smooth animations
- Interactive image viewing with zoom and pan capabilities
- Multi-threaded processing to keep UI responsive
- High-quality image dehazing using dark channel prior algorithm
- Side-by-side comparison of original and dehazed images
- Progress indicators for long-running operations

## Installation

1. Make sure you have Python 3.6+ installed
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:

```bash
python modern_dehaze_gui.py
```

2. Use the GUI:
   - Click "Load Image" to select an image for dehazing
   - The image will be automatically processed
   - Use mouse wheel to zoom in/out on either image
   - Click and drag to pan around when zoomed in
   - Click "Reset View" to return to the original view
   - Click "Save Result" to save the dehazed image

## Controls

- **Load Image**: Opens a file dialog to select an image
- **Save Result**: Saves the dehazed image to a location of your choice
- **Reset View**: Resets zoom and pan for both image viewers
- **Mouse Wheel**: Zoom in/out
- **Click and Drag**: Pan the image when zoomed in

## Technical Details

This application uses:
- PyQt5 for the modern UI components
- OpenCV for image processing
- Multi-threading for responsive UI during processing
- Custom image viewers with interactive controls

The dehazing algorithm implements the dark channel prior method with guided filter refinement and contrast enhancement.