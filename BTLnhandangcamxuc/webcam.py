import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Tải mô hình đã huấn luyện
model = load_model('emotion_recognition_model.h5')

# Định nghĩa các cảm xúc
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Khởi tạo webcam
cap = cv2.VideoCapture(0)  # 0 là chỉ số webcam mặc định

while True:
    # Đọc khung hình từ webcam
    ret, frame = cap.read()

    # Nếu không lấy được khung hình, tiếp tục vòng lặp
    if not ret:
        print("Không thể truy cập webcam!")
        break

    # Chuyển ảnh sang ảnh xám (grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Sử dụng Haar Cascade để phát hiện khuôn mặt ( scaleFactor: Hệ số giảm kích thước ảnh./ minNeighbors: Số lượng vùng lân cận cần thiết để xác nhận một đối tượng.)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Nếu phát hiện khuôn mặt, xử lý và dự đoán cảm xúc
    for (x, y, w, h) in faces:
        # Cắt khuôn mặt từ khung hình
        face = gray[y:y + h, x:x + w]

        # Resize ảnh mặt về kích thước phù hợp với mô hình (48x48) ( ảnh đã xử lys = Resize(ảnh gôốc,(48,48))
        face_resized = cv2.resize(face, (48, 48))

        # Chuẩn hóa và chuyển đổi ảnh thành dạng phù hợp với mô hình
        face_array = face_resized / 255.0
        face_array = np.expand_dims(face_array, axis=-1)  # Thêm kênh grayscale
        face_array = np.expand_dims(face_array, axis=0)  # Thêm batch dimension

        # Dự đoán cảm xúc (y^i = arg maxσ/j (zj) )
        prediction = model.predict(face_array)
        max_index = np.argmax(prediction)
        predicted_emotion = class_labels[max_index]

        # Vẽ hình chữ nhật quanh khuôn mặt và hiển thị cảm xúc dự đoán
        color = (0, 255, 0)  # Màu xanh lá cho khung mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Hiển thị khung hình với dự đoán cảm xúc
    cv2.imshow('Emotion Recognition', frame)

    # Thoát khỏi vòng lặp nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng webcam và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
