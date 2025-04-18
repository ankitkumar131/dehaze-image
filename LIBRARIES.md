# Libraries Documentation

## Core Libraries

### 1. OpenCV (cv2)
- **Purpose**: Primary image processing library
- **Usage**: 
  - Image loading and color space conversions (BGR, LAB)
  - Implementation of guided filter for transmission map refinement
  - Basic image operations (erosion, box filtering)
  - Adaptive histogram equalization for contrast enhancement
- **Key Functions**:
  - `cv2.cvtColor()`: Color space conversions
  - `cv2.boxFilter()`: For guided filtering
  - `cv2.createCLAHE()`: Contrast enhancement
  - `cv2.erode()`: Dark channel computation

### 2. NumPy (np)
- **Purpose**: Numerical computations and array operations
- **Usage**:
  - Efficient array operations for image processing
  - Mathematical computations in dehazing algorithm
  - Array manipulations and type conversions
- **Key Operations**:
  - Array operations (min, max, clip)
  - Type conversions (float32, uint8)
  - Array reshaping and indexing

### 3. Tkinter (tk)
- **Purpose**: Graphical User Interface (GUI) framework
- **Usage**:
  - Main application window and widgets
  - File dialogs for image selection
  - Message boxes for user notifications
- **Components**:
  - `filedialog`: File selection dialogs
  - `messagebox`: User notifications
  - `ttk`: Themed widgets

### 4. PIL (Python Imaging Library)
- **Purpose**: Image handling for GUI display
- **Usage**:
  - Converting between OpenCV and Tkinter-compatible image formats
  - Image resizing and display in GUI
- **Key Classes**:
  - `Image`: Basic image handling
  - `ImageTk`: Tkinter-compatible image objects

### 5. Matplotlib
- **Purpose**: Advanced visualization capabilities
- **Usage**:
  - Image display in the GUI
  - Integration with Tkinter for interactive visualization
- **Components**:
  - `FigureCanvasTkAgg`: Embedding plots in Tkinter
  - `NavigationToolbar2Tk`: Interactive navigation tools

## Integration Flow
1. Images are loaded and processed using OpenCV
2. NumPy handles the mathematical operations in the dehazing algorithm
3. PIL converts processed images for Tkinter display
4. Tkinter provides the user interface
5. Matplotlib enables advanced visualization features

This combination of libraries creates a robust image dehazing application with a user-friendly interface and efficient processing capabilities.