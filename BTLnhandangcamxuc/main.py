import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import numpy as np
import matplotlib.pyplot as plt

# Đường dẫn đến tập dữ liệu FER-2013
train_dir = 'nhandangcamxuc/train'  # Thay bằng đường dẫn thư mục train
val_dir = 'nhandangcamxuc/test'  # Thay bằng đường dẫn thư mục validation

# Sử dụng ImageDataGenerator để tiền xử lý ảnh
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Chuẩn hóa ảnh về [0, 1] ( pixel/norm = pixeloriginal/ 255.0)
    rotation_range=20,  # Xoay ngẫu nhiên trong khoảng 20 độ
    width_shift_range=0.2,  # Dịch ngang ảnh
    height_shift_range=0.2,  # Dịch dọc ảnh
    zoom_range=0.2,  # Phóng to/thu nhỏ ảnh
    horizontal_flip=True,  # Lật ngang ảnh
    fill_mode='nearest'  # Điền giá trị trống
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Chuẩn bị dữ liệu cho train và validation
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),  # Kích thước ảnh (chuẩn FER-2013)
    batch_size=64,
    class_mode='categorical',  # Phân loại nhiều lớp
    color_mode='grayscale'  # Ảnh xám
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical',
    color_mode='grayscale'
)

# Xây dựng mô hình CNN(
model = Sequential([
    # Convolutional layers ( Hàm kích hoạt)
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)), #( f(x)= max(0,x)
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    # Fully connected layers
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 lớp cảm xúc: Angry, Happy, Sad, ...
])

model.summary()

# Biên dịch mô hình ( Accuracy =so luong du đoan dung/ tống mau)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',#(Loss= - 1/n 1∑i=1 c∑j=1 yij log(y^ij)
    metrics=['accuracy']
)

# Huấn luyện mô hình
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=val_generator
)

# Đánh giá mô hình
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Lưu mô hình
model.save('emotion_recognition_model.h5')

# Vẽ biểu đồ Accuracy và Loss
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
