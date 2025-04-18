import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, 
                             QProgressBar, QSlider, QSplitter, QMessageBox,
                             QFrame, QGraphicsDropShadowEffect, QScrollArea)
from PyQt5.QtGui import QPixmap, QImage, QColor, QPalette, QFont, QIcon, QCursor, QPainter
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QPropertyAnimation, QEasingCurve

# Dehazing functions directly implemented from testr.py
# Function to apply guided filter for transmission map refinement
def guided_filter(guide, src, radius, eps):
    mean_guide = cv2.boxFilter(guide, -1, (radius, radius))
    mean_src = cv2.boxFilter(src, -1, (radius, radius))
    mean_guide_src = cv2.boxFilter(guide * src, -1, (radius, radius))
    cov_guide_src = mean_guide_src - mean_guide * mean_src

    mean_guide_guide = cv2.boxFilter(guide * guide, -1, (radius, radius))
    var_guide = mean_guide_guide - mean_guide * mean_guide

    a = cov_guide_src / (var_guide + eps)
    b = mean_src - a * mean_guide

    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))

    return mean_a * guide + mean_b

# Function to enhance contrast using adaptive histogram equalization
def enhance_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

# Function to get the dark channel prior
def get_dark_channel(img, patch_size):
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel

# Function to get the atmospheric light
def get_atmospheric_light(img, dark_channel, top_percent=0.1):
    img_size = img.shape[0] * img.shape[1]
    num_pixels = int(max(img_size * top_percent, 1))
    flat_img = img.reshape(img_size, 3)
    flat_dark = dark_channel.reshape(img_size)
    indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
    top_pixels = flat_img[indices]
    atmospheric_light = np.max(top_pixels, axis=0)
    return atmospheric_light

# Function to get the transmission
def get_transmission(img, atmospheric_light, omega=0.95, patch_size=15):
    normalized_img = img / atmospheric_light
    dark_channel = get_dark_channel(normalized_img, patch_size)
    transmission = 1 - omega * dark_channel
    return transmission

# Function to dehaze the image using the dark channel prior method with improvements
def dehaze_image(img, patch_size=15, omega=0.95):
    img = img.astype(np.float32) / 255.0
    
    # Adaptive patch size based on image size
    h, w = img.shape[:2]
    patch_size = min(max(int(min(h, w) * 0.02), 7), 15)
    
    dark_channel = get_dark_channel(img, patch_size)
    atmospheric_light = get_atmospheric_light(img, dark_channel)
    transmission = get_transmission(img, atmospheric_light, omega, patch_size)
    
    # Refine transmission map using guided filter
    gray_img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    transmission = guided_filter(gray_img.astype(np.float32) / 255.0, 
                               transmission, radius=40, eps=1e-3)
    
    # Ensure minimum transmission to preserve contrast
    transmission = np.maximum(transmission, 0.1)
    
    # Dehaze the image
    dehazed = np.empty(img.shape, img.dtype)
    for i in range(3):
        dehazed[:, :, i] = (img[:, :, i] - atmospheric_light[i]) / transmission + atmospheric_light[i]
    
    # Clip and convert to uint8
    dehazed = np.clip(dehazed, 0, 1)
    dehazed = (dehazed * 255).astype(np.uint8)
    
    # Enhance contrast and color
    dehazed = enhance_contrast(dehazed)

    return dehazed

# Worker thread for processing images without freezing the UI
class DehazingThread(QThread):
    finished = pyqtSignal(np.ndarray)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    
    def __init__(self, image):
        super().__init__()
        self.image = image
        
    def run(self):
        try:
            # Update progress at different stages
            self.progress.emit(10)
            # Process the image using the dehazing algorithm
            result = dehaze_image(self.image.copy())
            self.progress.emit(100)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

# Custom image viewer with zoom and pan capabilities
class ImageViewer(QLabel):
    # Add signals for synchronization
    zoomChanged = pyqtSignal(float)
    panChanged = pyqtSignal(int, int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        # Increase height to 450 pixels (from 375)
        self.setMinimumSize(400, 850)
        self.setStyleSheet("background-color: #1E1E2E; border-radius: 10px;")
        self.setScaledContents(False)
        
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 0)
        self.setGraphicsEffect(shadow)
        
        # Initialize variables for zoom and pan
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.panning = False
        self.last_pos = None
        self.original_pixmap = None
        self._sync_in_progress = False
        
    def setImage(self, image):
        if image is None:
            return
            
        if isinstance(image, np.ndarray):
            # Convert OpenCV image (BGR) to QPixmap
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            q_img = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
        else:
            pixmap = QPixmap(image)
            
        self.original_pixmap = pixmap
        self.updatePixmap()
        
    def updatePixmap(self):
        if self.original_pixmap is None:
            return
            
        # Calculate the scaled size
        scaled_size = self.original_pixmap.size() * self.zoom_factor
        scaled_pixmap = self.original_pixmap.scaled(scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Create a new pixmap with the size of the label
        visible_pixmap = QPixmap(self.size())
        visible_pixmap.fill(Qt.transparent)
        
        # Calculate the position to draw the scaled pixmap
        x = int((self.width() - scaled_pixmap.width()) / 2 + self.pan_x)
        y = int((self.height() - scaled_pixmap.height()) / 2 + self.pan_y)
        
        # Draw the scaled pixmap onto the visible pixmap
        painter = QPainter(visible_pixmap)
        painter.drawPixmap(x, y, scaled_pixmap)
        painter.end()
        
        # Store the pixmap temporarily and set it after the painter is destroyed
        temp_pixmap = QPixmap(visible_pixmap)
        self.setPixmap(temp_pixmap)
        
    def wheelEvent(self, event):
        if not self._sync_in_progress:
            # Zoom in/out with mouse wheel
            zoom_in_factor = 1.1
            zoom_out_factor = 1 / zoom_in_factor
            
            if event.angleDelta().y() > 0:
                self.zoom_factor *= zoom_in_factor
            else:
                self.zoom_factor *= zoom_out_factor
                
            # Limit zoom range
            self.zoom_factor = max(0.1, min(5.0, self.zoom_factor))
            
            self.updatePixmap()
            # Emit signal for synchronization
            self.zoomChanged.emit(self.zoom_factor)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.panning = True
            self.last_pos = event.pos()
            self.setCursor(QCursor(Qt.ClosedHandCursor))
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.panning = False
            self.setCursor(QCursor(Qt.ArrowCursor))
            
    def mouseMoveEvent(self, event):
        if not self._sync_in_progress and self.panning and self.last_pos is not None:
            delta = event.pos() - self.last_pos
            self.pan_x += delta.x()
            self.pan_y += delta.y()
            self.last_pos = event.pos()
            self.updatePixmap()
            # Emit signal for synchronization
            self.panChanged.emit(self.pan_x, self.pan_y)
            
    def resetView(self):
        if not self._sync_in_progress:
            self.zoom_factor = 1.0
            self.pan_x = 0
            self.pan_y = 0
            self.updatePixmap()
            # Emit signals for synchronization
            self.zoomChanged.emit(self.zoom_factor)
            self.panChanged.emit(self.pan_x, self.pan_y)
            
    def syncZoom(self, zoom_factor):
        if not self._sync_in_progress:
            self._sync_in_progress = True
            self.zoom_factor = zoom_factor
            self.updatePixmap()
            self._sync_in_progress = False
            
    def syncPan(self, x, y):
        if not self._sync_in_progress:
            self._sync_in_progress = True
            self.pan_x = x
            self.pan_y = y
            self.updatePixmap()
            self._sync_in_progress = False

# Custom button with hover effects
class StyledButton(QPushButton):
    def __init__(self, text, parent=None, icon=None):
        super().__init__(text, parent)
        self.setFont(QFont("Segoe UI", 10))
        self.setCursor(Qt.PointingHandCursor)
        
        # Set minimum size
        self.setMinimumSize(120, 40)
        
        # Set icon if provided
        if icon:
            self.setIcon(QIcon(icon))
            self.setIconSize(QSize(20, 20))
        
        # Apply stylesheet
        self.setStyleSheet("""
            QPushButton {
                background-color: #7E57C2;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #9575CD;
            }
            QPushButton:pressed {
                background-color: #673AB7;
            }
            QPushButton:disabled {
                background-color: #B39DDB;
                color: #E0E0E0;
            }
        """)

# Main application window
class ImageDehazingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.original_image = None
        self.dehazed_image = None
        self.initUI()
        
    def initUI(self):
        # Set window properties
        self.setWindowTitle("Advanced Image Dehazing Tool")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet("background-color: #121212;")
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Create header with title
        header = QLabel("Advanced Image Dehazing Tool")
        header.setAlignment(Qt.AlignCenter)
        header.setFont(QFont("Segoe UI", 24, QFont.Bold))
        header.setStyleSheet("color: #E0E0E0; margin-bottom: 10px;")
        main_layout.addWidget(header)
        
        # Create button layout
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        # Create buttons
        self.load_button = StyledButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        
        self.save_button = StyledButton("Save Result")
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setEnabled(False)
        
        self.reset_button = StyledButton("Reset View")
        self.reset_button.clicked.connect(self.reset_views)
        self.reset_button.setEnabled(False)
        
        # Add buttons to layout
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addStretch(1)
        
        # Add button layout to main layout
        main_layout.addLayout(button_layout)
        
        # Create progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 5px;
                background-color: #424242;
                height: 10px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #7E57C2;
                border-radius: 5px;
            }
        """)
        self.progress_bar.hide()
        main_layout.addWidget(self.progress_bar)
        
        # Create image viewers container
        image_container = QSplitter(Qt.Horizontal)
        
        # Create image viewers
        self.original_viewer = ImageViewer()
        self.dehazed_viewer = ImageViewer()
        
        # Connect signals for synchronization
        self.original_viewer.zoomChanged.connect(self.dehazed_viewer.syncZoom)
        self.original_viewer.panChanged.connect(self.dehazed_viewer.syncPan)
        self.dehazed_viewer.zoomChanged.connect(self.original_viewer.syncZoom)
        self.dehazed_viewer.panChanged.connect(self.original_viewer.syncPan)
        
        # Create labels for the viewers
        original_container = QWidget()
        original_layout = QVBoxLayout(original_container)
        original_label = QLabel("Original Image")
        original_label.setAlignment(Qt.AlignCenter)
        original_label.setFont(QFont("Segoe UI", 12))
        original_label.setStyleSheet("color: #E0E0E0; margin-bottom: 5px;")
        original_layout.addWidget(original_label)
        original_layout.addWidget(self.original_viewer)
        
        dehazed_container = QWidget()
        dehazed_layout = QVBoxLayout(dehazed_container)
        dehazed_label = QLabel("Dehazed Image")
        dehazed_label.setAlignment(Qt.AlignCenter)
        dehazed_label.setFont(QFont("Segoe UI", 12))
        dehazed_label.setStyleSheet("color: #E0E0E0; margin-bottom: 5px;")
        dehazed_layout.addWidget(dehazed_label)
        dehazed_layout.addWidget(self.dehazed_viewer)
        
        # Add containers to splitter
        image_container.addWidget(original_container)
        image_container.addWidget(dehazed_container)
        image_container.setSizes([600, 600])
        
        # Add image container to main layout
        main_layout.addWidget(image_container, 1)
        
        # Create status bar
        self.statusBar().setStyleSheet("color: #E0E0E0; background-color: #212121;")
        self.statusBar().showMessage("Ready")
        
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        
        if file_path:
            try:
                # Load image using OpenCV
                self.original_image = cv2.imread(file_path)
                if self.original_image is None:
                    raise Exception("Failed to load image")
                    
                # Display original image
                self.original_viewer.setImage(self.original_image)
                
                # Show progress bar and start processing
                self.progress_bar.setValue(0)
                self.progress_bar.show()
                self.statusBar().showMessage("Processing image...")
                
                # Start dehazing in a separate thread
                self.dehazing_thread = DehazingThread(self.original_image)
                self.dehazing_thread.progress.connect(self.update_progress)
                self.dehazing_thread.finished.connect(self.display_dehazed_image)
                self.dehazing_thread.error.connect(self.show_error)
                self.dehazing_thread.start()
                
                # Enable reset button
                self.reset_button.setEnabled(True)
                
            except Exception as e:
                self.show_error(str(e))
                
    def update_progress(self, value):
        self.progress_bar.setValue(value)
        
    def display_dehazed_image(self, image):
        self.dehazed_image = image
        self.dehazed_viewer.setImage(self.dehazed_image)
        self.progress_bar.hide()
        self.save_button.setEnabled(True)
        self.statusBar().showMessage("Image processed successfully")
        
    def save_image(self):
        if self.dehazed_image is None:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )
        
        if file_path:
            try:
                # Save image using OpenCV
                cv2.imwrite(file_path, self.dehazed_image)
                self.statusBar().showMessage(f"Image saved to {file_path}")
            except Exception as e:
                self.show_error(f"Error saving image: {str(e)}")
                
    def reset_views(self):
        self.original_viewer.resetView()
        self.dehazed_viewer.resetView()
        
    def show_error(self, message):
        self.progress_bar.hide()
        QMessageBox.critical(self, "Error", message)
        self.statusBar().showMessage("Error: " + message)

# Main entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create dark palette
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(18, 18, 18))
    dark_palette.setColor(QPalette.WindowText, QColor(224, 224, 224))
    dark_palette.setColor(QPalette.Base, QColor(30, 30, 46))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(224, 224, 224))
    dark_palette.setColor(QPalette.ToolTipText, QColor(224, 224, 224))
    dark_palette.setColor(QPalette.Text, QColor(224, 224, 224))
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, QColor(224, 224, 224))
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(126, 87, 194))
    dark_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    
    # Apply palette
    app.setPalette(dark_palette)
    
    # Create and show main window
    window = ImageDehazingApp()
    window.show()
    
    sys.exit(app.exec_())