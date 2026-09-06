import cv2

cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam opened successfully. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to grab frame.")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()