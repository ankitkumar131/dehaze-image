import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# Function to update the image display

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

# Function to update the image display
def update_image_display():
    global original_image, dehazed_image, panel, canvas, toolbar
    if original_image is not None and dehazed_image is not None:
        # Create image display container
        if canvas is not None:
            canvas.get_tk_widget().destroy()
        if toolbar is not None:
            toolbar.destroy()
            
        # Create a figure with modern styling
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), facecolor='#2E3440')
        fig.patch.set_facecolor('#2E3440')
        
        # Display the original image with enhanced styling
        ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image', color='#ECEFF4', pad=20, fontsize=12)
        ax1.axis('off')
        
        # Display the dehazed image with enhanced styling
        ax2.imshow(cv2.cvtColor(dehazed_image, cv2.COLOR_BGR2RGB))
        ax2.set_title('Dehazed Image', color='#ECEFF4', pad=20, fontsize=12)
        ax2.axis('off')
        
        # Add spacing between subplots
        plt.tight_layout(pad=3.0)
        
        # Create image display frame
        display_frame = tk.Frame(main_container, bg='#2E3440')
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add zoom functionality with styled canvas
        canvas = FigureCanvasTkAgg(fig, master=display_frame)
        canvas.draw()
        
        # Style the toolbar
        toolbar = NavigationToolbar2Tk(canvas, display_frame, pack_toolbar=False)
        toolbar.config(background='#2E3440')
        for button in toolbar.winfo_children():
            if isinstance(button, tk.Button):
                button.config(background='#5E81AC', activebackground='#81A1C1')
        toolbar.update()
        
        # Synchronize zoom and pan between subplots with recursion prevention
        sync_in_progress = {'x': False, 'y': False}
        
        def on_xlims_change(event_ax):
            if not sync_in_progress['x']:
                try:
                    sync_in_progress['x'] = True
                    if event_ax == ax1:
                        ax2.set_xlim(ax1.get_xlim())
                    else:
                        ax1.set_xlim(ax2.get_xlim())
                    canvas.draw_idle()
                finally:
                    sync_in_progress['x'] = False

        def on_ylims_change(event_ax):
            if not sync_in_progress['y']:
                try:
                    sync_in_progress['y'] = True
                    if event_ax == ax1:
                        ax2.set_ylim(ax1.get_ylim())
                    else:
                        ax1.set_ylim(ax2.get_ylim())
                    canvas.draw_idle()
                finally:
                    sync_in_progress['y'] = False

        ax1.callbacks.connect('xlim_changed', lambda event: on_xlims_change(ax1))
        ax2.callbacks.connect('xlim_changed', lambda event: on_xlims_change(ax2))
        ax1.callbacks.connect('ylim_changed', lambda event: on_ylims_change(ax1))
        ax2.callbacks.connect('ylim_changed', lambda event: on_ylims_change(ax2))
        
        # Pack the canvas and toolbar with proper spacing
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        toolbar.pack(side=tk.BOTTOM, fill=tk.X, padx=10)
        
        # Clear the figure for the next update
        plt.close(fig)

# Function to browse and load the image
def load_image():
    global original_image, dehazed_image, panel, canvas, toolbar
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")])
    if path:
        # Show loading indicator
        loading_label = tk.Label(main_container, 
                                text="Processing image...", 
                                font=("Helvetica", 12),
                                bg="#2E3440",
                                fg="#ECEFF4")
        loading_label.pack(pady=20)
        root.update()
        
        try:
            original_image = cv2.imread(path)
            if original_image is None:
                messagebox.showerror("Error", "Failed to load image. Please try another file.")
                loading_label.destroy()
                return
                
            dehazed_image = dehaze_image(original_image.copy())
            
            # Clean up previous display components
            for widget in main_container.winfo_children():
                if isinstance(widget, tk.Frame) and widget not in [header_frame, button_frame]:
                    widget.destroy()
            
            if panel is not None:
                panel.destroy()
            if canvas is not None:
                canvas.get_tk_widget().destroy()
            if toolbar is not None:
                toolbar.destroy()
                
            update_image_display()
            messagebox.showinfo("Success", "Image processed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        finally:
            loading_label.destroy()

# Function to save the dehazed image
def save_image():
    global dehazed_image
    if dehazed_image is not None:
        path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")])
        if path:
            cv2.imwrite(path, cv2.cvtColor(dehazed_image, cv2.COLOR_RGB2BGR))
            messagebox.showinfo("Success", "Image saved successfully.")

# Initialize the original and dehazed images
original_image = None
dehazed_image = None
panel = None
canvas = None
toolbar = None

# Set up custom styles and theme
def setup_theme():
    # Define colors
    bg_color = '#2E3440'
    fg_color = '#ECEFF4'
    button_bg = '#5E81AC'
    button_hover_bg = '#81A1C1'
    
    # Configure root window style
    root.configure(bg=bg_color)
    
    # Configure button style
    style = tk.ttk.Style()
    style.theme_use('clam')
    
    # Configure custom button style
    style.configure('Custom.TButton',
                    background=button_bg,
                    foreground=fg_color,
                    padding=(20, 10),
                    font=('Helvetica', 10),
                    borderwidth=0)
    style.map('Custom.TButton',
              background=[('active', button_hover_bg)],
              foreground=[('active', fg_color)])
    
    return bg_color, fg_color

# Create the main window
root = tk.Tk()
root.title("Image Dehazing Tool")
root.geometry('1200x800')

# Apply theme
bg_color, fg_color = setup_theme()

# Create main container
main_container = tk.Frame(root, bg=bg_color)
main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# Create header frame
header_frame = tk.Frame(main_container, bg=bg_color)
header_frame.pack(fill=tk.X, pady=(0, 20))

# Add title
title_label = tk.Label(header_frame, 
                      text="Image Dehazing Tool", 
                      font=('Helvetica', 24, 'bold'),
                      bg=bg_color,
                      fg=fg_color)
title_label.pack()

# Create button frame with modern styling
button_frame = tk.Frame(main_container, bg=bg_color)
button_frame.pack(fill=tk.X, pady=(0, 20))

# Create and place the buttons with modern styling
load_button = tk.ttk.Button(button_frame, 
                          text="Load Image", 
                          command=load_image,
                          style='Custom.TButton')
load_button.pack(side=tk.LEFT, padx=5)

save_button = tk.ttk.Button(button_frame, 
                          text="Save Dehazed Image", 
                          command=save_image,
                          style='Custom.TButton')
save_button.pack(side=tk.LEFT, padx=5)

# Start the GUI loop
root.mainloop()