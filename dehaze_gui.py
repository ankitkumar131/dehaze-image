import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

class DehazingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Dehazing Application")
        self.root.geometry("1200x800")
        
        # Load the dehazing model
        try:
            self.model = load_model('dehazing_model.h5')
        except:
            messagebox.showerror("Error", "Could not load the dehazing model!")
            root.destroy()
            return
        
        self.setup_gui()
        self.original_image = None
        self.dehazed_image = None
        
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Load image button
        self.load_btn = ttk.Button(button_frame, text="Load Image", command=self.load_image)
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        # Dehaze button
        self.dehaze_btn = ttk.Button(button_frame, text="Dehaze Image", command=self.dehaze_image)
        self.dehaze_btn.pack(side=tk.LEFT, padx=5)
        self.dehaze_btn.config(state='disabled')
        
        # Save button
        self.save_btn = ttk.Button(button_frame, text="Save Dehazed Image", command=self.save_image)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        self.save_btn.config(state='disabled')
        
        # Image frames
        image_container = ttk.Frame(main_frame)
        image_container.pack(fill=tk.BOTH, expand=True)
        
        # Original image frame
        original_frame = ttk.LabelFrame(image_container, text="Original Image")
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.original_label = ttk.Label(original_frame)
        self.original_label.pack(fill=tk.BOTH, expand=True)
        
        # Dehazed image frame
        dehazed_frame = ttk.LabelFrame(image_container, text="Dehazed Image")
        dehazed_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.dehazed_label = ttk.Label(dehazed_frame)
        self.dehazed_label.pack(fill=tk.BOTH, expand=True)
        
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")
        ])
        
        if file_path:
            try:
                # Load and display original image
                self.original_image = cv2.imread(file_path)
                if self.original_image is None:
                    raise Exception("Could not load image")
                
                # Convert BGR to RGB for display
                rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                self.display_image(rgb_image, self.original_label)
                
                # Enable dehaze button
                self.dehaze_btn.config(state='normal')
                # Disable save button until dehazing is done
                self.save_btn.config(state='disabled')
                
            except Exception as e:
                messagebox.showerror("Error", f"Error loading image: {str(e)}")
    
    def dehaze_image(self):
        if self.original_image is None:
            return
        
        try:
            print("Starting dehazing process...")
            # Prepare image for model
            input_image = cv2.resize(self.original_image, (256, 256))
            print(f"Input image shape after resize: {input_image.shape}")
            
            # Convert BGR to RGB for model input
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            print("Converted to RGB for model input")
            
            # Normalize the image
            input_image = input_image.astype('float32') / 255.0
            input_image = np.expand_dims(input_image, axis=0)
            print(f"Normalized input shape: {input_image.shape}")
            
            # Process image through model
            print("Running model prediction...")
            dehazed = self.model.predict(input_image)
            print(f"Model output shape: {dehazed.shape}")
            
            # Post-process output
            dehazed = dehazed[0]  # Remove batch dimension
            dehazed = np.clip(dehazed * 255.0, 0, 255).astype(np.uint8)
            print(f"Post-processed output shape: {dehazed.shape}")
            
            # Resize back to original dimensions
            dehazed = cv2.resize(dehazed, (self.original_image.shape[1], self.original_image.shape[0]))
            print(f"Resized output shape: {dehazed.shape}")
            
            # Store dehazed image in BGR format for saving
            self.dehazed_image = cv2.cvtColor(dehazed, cv2.COLOR_RGB2BGR)
            
            # Display dehazed image (keep in RGB format for display)
            print("Displaying dehazed image...")
            self.display_image(dehazed, self.dehazed_label)
            
            # Enable save button
            self.save_btn.config(state='normal')
            print("Dehazing process completed successfully")
            
        except Exception as e:
            print(f"Error during dehazing: {str(e)}")
            messagebox.showerror("Error", f"Error during dehazing: {str(e)}")
    
    def save_image(self):
        if self.dehazed_image is None:
            return
            
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG files", "*.png")])
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.dehazed_image)
                messagebox.showinfo("Success", "Dehazed image saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving image: {str(e)}")
    
    def display_image(self, image, label):
        # Calculate aspect ratio
        aspect_ratio = image.shape[1] / image.shape[0]
        
        # Calculate new dimensions while maintaining aspect ratio
        max_width = 550
        max_height = 700
        
        if aspect_ratio > 1:
            # Width is larger
            new_width = min(image.shape[1], max_width)
            new_height = int(new_width / aspect_ratio)
        else:
            # Height is larger
            new_height = min(image.shape[0], max_height)
            new_width = int(new_height * aspect_ratio)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height))
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(Image.fromarray(resized))
        
        # Update label
        label.configure(image=photo)
        label.image = photo  # Keep a reference!

if __name__ == "__main__":
    root = tk.Tk()
    app = DehazingGUI(root)
    root.mainloop()