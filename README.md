# Image Dehazing Tool

A sophisticated image dehazing application that implements the Dark Channel Prior method with several improvements for removing haze from images. The application features a modern, user-friendly GUI built with tkinter and provides real-time image processing capabilities.

## Features

- Advanced image dehazing using Dark Channel Prior algorithm
- Real-time image processing with progress indication
- Interactive GUI with side-by-side image comparison
- Zoom and pan synchronization between original and dehazed images
- Modern dark theme interface
- Support for multiple image formats (JPG, PNG, BMP)
- Enhanced contrast using adaptive histogram equalization
- Guided filter for transmission map refinement

## Libraries Used

### Core Image Processing

1. **OpenCV (cv2)**
   - Used for fundamental image processing operations
   - Handles image loading, saving, and color space conversions
   - Implements core algorithms like erode, boxFilter
   - Provides CLAHE (Contrast Limited Adaptive Histogram Equalization)

2. **NumPy (np)**
   - Enables efficient array operations for image manipulation
   - Handles mathematical computations in the dehazing algorithm
   - Provides optimized array operations for dark channel calculation
   - Used for atmospheric light estimation

### GUI Framework

3. **Tkinter (tk)**
   - Main GUI framework for the application
   - Provides window management and widget layout
   - Handles file dialogs and message boxes
   - Implements custom styling and theming

4. **PIL (Python Imaging Library)**
   - Bridges between OpenCV images and Tkinter display
   - Handles image format conversions
   - Provides ImageTk for displaying images in the GUI

5. **Matplotlib**
   - Creates interactive image display with zoom capabilities
   - Implements side-by-side image comparison
   - Provides navigation toolbar for image interaction
   - Handles synchronized zooming and panning

## Technical Implementation

### Dehazing Algorithm

The application implements an enhanced version of the Dark Channel Prior method with the following components:

1. **Dark Channel Calculation**
   - Adaptive patch size based on image dimensions
   - Efficient implementation using OpenCV's erode operation

2. **Atmospheric Light Estimation**
   - Uses top percentile of brightest pixels
   - Implements efficient selection using NumPy's argpartition

3. **Transmission Map Estimation**
   - Calculates initial transmission map using dark channel
   - Applies guided filter for refinement
   - Implements minimum transmission threshold

4. **Image Recovery**
   - Recovers haze-free image using the radiometric model
   - Applies contrast enhancement using CLAHE
   - Implements color preservation techniques

### GUI Features

1. **Modern Interface**
   - Dark theme implementation using custom styles
   - Responsive layout with proper padding and spacing
   - Interactive buttons with hover effects

2. **Image Display**
   - Side-by-side comparison of original and dehazed images
   - Synchronized zoom and pan functionality
   - Interactive navigation toolbar
   - Loading indicator during processing

3. **File Operations**
   - Support for multiple image formats
   - File dialogs for loading and saving images
   - Error handling for file operations

## Usage

1. Launch the application
2. Click "Load Image" to select an image file
3. Wait for the dehazing process to complete
4. Use the navigation toolbar to zoom and pan both images
5. Click "Save Dehazed Image" to save the processed image

## Error Handling

The application implements comprehensive error handling for:
- Image loading failures
- Processing errors
- File saving issues
- Invalid file formats

## Performance Considerations

- Adaptive patch size selection based on image dimensions
- Efficient array operations using NumPy
- Optimized guided filter implementation
- Progress indication for long-running operations



# Image Dehazing Tool

A sophisticated image dehazing application that implements the Dark Channel Prior method with several improvements for effective haze removal from images. This tool provides a modern, user-friendly GUI interface for image processing.

## Features

- **Advanced Dehazing Algorithm**: Implements an enhanced version of the Dark Channel Prior method
- **Adaptive Processing**: Automatically adjusts parameters based on image characteristics
- **Interactive GUI**: Modern, dark-themed interface with synchronized image comparison
- **Real-time Processing**: Immediate visual feedback with loading indicators
- **Image Enhancement**: Includes contrast enhancement using adaptive histogram equalization
- **Zoom Synchronization**: Synchronized zooming and panning between original and dehazed images

## Technical Implementation

### Core Components

1. **Dark Channel Prior Algorithm**
   - Calculates dark channel using adaptive patch sizes
   - Optimized atmospheric light estimation
   - Transmission map computation with refinement

2. **Image Enhancement**
   - Guided filter for transmission map refinement
   - Adaptive histogram equalization for contrast enhancement
   - Color preservation techniques

3. **GUI Features**
   - Modern dark theme using custom styling
   - Interactive matplotlib integration
   - Synchronized view controls
   - Progress indicators

### Key Functions

- `dehaze_image()`: Main dehazing function with adaptive parameter selection
- `guided_filter()`: Implements edge-preserving smoothing
- `enhance_contrast()`: Applies CLAHE for better visibility
- `get_dark_channel()`: Calculates dark channel prior
- `get_atmospheric_light()`: Estimates atmospheric light
- `get_transmission()`: Computes transmission map

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- Matplotlib
- Tkinter
- Pillow (PIL)

## Usage

1. Run the application:
   ```bash
   python testr.py
   ```

2. Use the GUI:
   - Click "Load Image" to select an image for dehazing
   - Wait for processing to complete
   - Use zoom and pan controls to examine details
   - Click "Save Dehazed Image" to save the result

## Algorithm Parameters

- Patch Size: Adaptive (2% of min image dimension, bounded between 7 and 15)
- Omega (Dehazing Strength): 0.95
- Guided Filter: radius=40, eps=1e-3
- CLAHE: clipLimit=2.0, tileGridSize=(8,8)

## Implementation Details

### Dehazing Process

1. **Image Preprocessing**
   - Conversion to float32 format
   - Normalization to [0,1] range

2. **Dark Channel Computation**
   - Adaptive patch size selection
   - Minimum channel calculation
   - Morphological erosion

3. **Atmospheric Light Estimation**
   - Top 0.1% brightest pixels selection
   - Maximum intensity calculation

4. **Transmission Map**
   - Initial estimation using dark channel
   - Refinement using guided filter
   - Minimum transmission threshold

5. **Image Recovery**
   - Scene radiance recovery
   - Post-processing enhancement

### GUI Implementation

- Custom themed interface using ttk styles
- Matplotlib integration for image display
- Synchronized zooming and panning
- Progress indicators for long operations

## Performance Considerations

- Adaptive parameter selection based on image size
- Efficient numpy operations for matrix calculations
- Guided filter optimization for edge preservation
- Memory-efficient image processing

## Future Improvements

- Batch processing capability
- Additional image enhancement options
- Performance optimization for larger images
- Support for video dehazing
- Custom parameter adjustment interface