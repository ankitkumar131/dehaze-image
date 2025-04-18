import os
import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Bridge
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    
    # Decoder
    up5 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3], axis=3)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(up5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)
    
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2], axis=3)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)
    
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1], axis=3)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)
    
    outputs = Conv2D(3, 1, activation='sigmoid')(conv7)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')
    return model

def load_and_preprocess_image(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    return img / 255.0

def load_dataset(hazy_dir, clear_dir, target_size=(256, 256)):
    hazy_images = []
    clear_images = []
    
    hazy_files = sorted([f for f in os.listdir(hazy_dir) if f.endswith('.png')])
    clear_files = sorted([f for f in os.listdir(clear_dir) if f.endswith('.png')])
    
    for hazy_file, clear_file in zip(hazy_files, clear_files):
        hazy_path = os.path.join(hazy_dir, hazy_file)
        clear_path = os.path.join(clear_dir, clear_file)
        
        hazy_img = load_and_preprocess_image(hazy_path, target_size)
        clear_img = load_and_preprocess_image(clear_path, target_size)
        
        hazy_images.append(hazy_img)
        clear_images.append(clear_img)
    
    return np.array(hazy_images), np.array(clear_images)

def create_data_generators(X_train, y_train):
    # Split data into training and validation sets
    val_split = 0.2
    split_idx = int(len(X_train) * (1 - val_split))
    
    X_train_split = X_train[:split_idx]
    y_train_split = y_train[:split_idx]
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]
    
    data_gen_args = dict(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    
    train_datagen = ImageDataGenerator(**data_gen_args)
    val_datagen = ImageDataGenerator()
    
    seed = 1
    batch_size = 8
    
    train_img_generator = train_datagen.flow(X_train_split, batch_size=batch_size, seed=seed)
    train_mask_generator = train_datagen.flow(y_train_split, batch_size=batch_size, seed=seed)
    
    val_img_generator = val_datagen.flow(X_val, batch_size=batch_size, seed=seed)
    val_mask_generator = val_datagen.flow(y_val, batch_size=batch_size, seed=seed)
    
    train_generator = zip(train_img_generator, train_mask_generator)
    val_generator = zip(val_img_generator, val_mask_generator)
    
    return train_generator, val_generator, len(X_train_split), len(X_val)

def train_model():
    # Set paths
    hazy_dir = 'smoke_images'
    clear_dir = 'no_smoke_images'
    
    # Load and preprocess dataset
    X_train, y_train = load_dataset(hazy_dir, clear_dir)
    
    # Create data generators for augmentation
    train_generator, val_generator, train_size, val_size = create_data_generators(X_train, y_train)
    
    # Create and compile model
    model = create_unet_model()
    
    # Set up callbacks
    checkpoint = ModelCheckpoint('dehazing_model.h5',
                                monitor='val_loss',
                                save_best_only=True,
                                mode='min')
    
    # Train model with 1 epoch
    model.fit(
        train_generator,
        steps_per_epoch=train_size // 8,
        epochs=1,
        validation_data=val_generator,
        validation_steps=val_size // 8,
        callbacks=[checkpoint])

if __name__ == '__main__':
    train_model()