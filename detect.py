"""
KrishiNova AI - Step 1: Real-Time Object Detection Prototype
---------------------------------------------------------------
Webcam -> YOLOv8n -> Bounding Boxes + Class Name + Confidence

Beginner-friendly, CPU-only, lightweight (suitable for low-spec laptops).
Press 'Q' to quit the webcam window.
"""

import cv2
from ultralytics import YOLO

# -----------------------------------------------------------------
# STEP 1: Load the lightweight pretrained YOLO model
# -----------------------------------------------------------------
# "yolov8n.pt" = YOLOv8 Nano -> smallest & fastest YOLOv8 model,
# ideal for CPU-only, low-RAM laptops like yours.
# The file is auto-downloaded ONCE (first run only) and cached
# locally in this folder / ultralytics cache for future runs.
print("Loading YOLOv8n model (first run may download the weights)...")
model = YOLO("yolov8n.pt")

# -----------------------------------------------------------------
# STEP 2: Open the laptop webcam
# -----------------------------------------------------------------
# 0 = default webcam. Change to 1 if you have an external camera.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam. Check that no other app is using it.")
    exit()

print("Webcam started. Press 'Q' in the window to quit.")

# -----------------------------------------------------------------
# STEP 3: Real-time detection loop
# -----------------------------------------------------------------
while True:
    ret, frame = cap.read()  # Read one frame from the webcam
    if not ret:
        print("ERROR: Failed to grab frame from webcam.")
        break

    # Run YOLO detection on the current frame.
    # conf=0.5  -> only show detections with confidence >= 50%
    # verbose=False -> keeps the terminal clean (no per-frame logs)
    results = model(frame, conf=0.5, verbose=False)

    # results[0].boxes contains all detected objects in this frame
    for box in results[0].boxes:
        # Get bounding box coordinates (top-left, bottom-right)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Get class id -> class name (e.g. "person", "car")
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # Get confidence score (0.0 - 1.0)
        confidence = float(box.conf[0])

        # Draw the bounding box (green rectangle)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Prepare label text: "class_name 0.87"
        label = f"{class_name} {confidence:.2f}"

        # Draw a filled background for the label so text is readable
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 0), -1)

        # Draw the label text
        cv2.putText(
            frame, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )

    # Show the live window with detections drawn on it
    cv2.imshow("KrishiNova AI - Step 1: YOLO Detection (Press Q to quit)", frame)

    # Wait 1ms for a key press; quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Quitting...")
        break

# -----------------------------------------------------------------
# STEP 4: Cleanup
# -----------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
