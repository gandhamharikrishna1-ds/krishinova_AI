import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam started.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break

    results = model.predict(
        frame,
        conf=0.5,
        verbose=False
    )

    annotated_frame = results[0].plot()

    cv2.imshow("KrishiNova - Webcam Test", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()