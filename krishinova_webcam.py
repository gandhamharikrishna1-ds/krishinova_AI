import time
from pathlib import Path

import cv2
from ultralytics import YOLO


THREAT_AUDIO = {
    "Bird": ("bird_alert.wav", "Freesound field fan-noise bird-scaring sound"),
    "Crow": ("bird_alert.wav", "Same bird-scaring sound used for Bird"),
    "Monkey": ("monkey_alert.wav", "Monkey troop alarm call"),
    "Elephant": ("elephant_alert.wav", "Elephant bee-threat alarm rumble"),
    "Wildboar": ("wildboar_alert.wav", "Dog-bark predator warning cue"),
    "Cow": ("cow_alert.wav", "Cowbell livestock movement cue"),
}

THREAT_PRIORITY = tuple(THREAT_AUDIO)


class AlertEngine:
    def __init__(self, cooldown_seconds=10):
        self.sounds_folder = Path(__file__).resolve().parent / "sounds"
        self.sounds_folder.mkdir(parents=True, exist_ok=True)
        self.cooldown_seconds = cooldown_seconds
        self.last_attempt_times = {}
        self.iot_message_until = 0

    def update(self, detections):
        now = time.monotonic()
        detected_names = {
            "Cow" if detection["name"] == "cow" else detection["name"]
            for detection in detections
            if detection["name"] in THREAT_AUDIO or detection["name"] == "cow"
        }
        detected_threats = [
            threat for threat in THREAT_PRIORITY if threat in detected_names
        ]

        if not detected_threats:
            return [], "None", "NO THREAT", "None", now < self.iot_message_until

        threat = detected_threats[0]
        sound_filename, attempt_description = THREAT_AUDIO[threat]
        previous_attempt = self.last_attempt_times.get(threat)

        if previous_attempt is not None and now - previous_attempt < self.cooldown_seconds:
            status = "AUDIO UNAVAILABLE" if sound_filename is None else "COOLDOWN"
        else:
            self.last_attempt_times[threat] = now
            self.iot_message_until = now + 3
            print(f"Threat detected: {threat}")
            print(f"Deterrent attempt: {attempt_description}")
            sound_path = self.sounds_folder / sound_filename if sound_filename else None

            if sound_path is None or not sound_path.is_file():
                status = "AUDIO UNAVAILABLE"
            else:
                try:
                    import winsound

                    winsound.PlaySound(
                        str(sound_path),
                        winsound.SND_FILENAME | winsound.SND_ASYNC,
                    )
                    status = "PLAYING"
                except (ImportError, RuntimeError, OSError):
                    status = "AUDIO UNAVAILABLE"

            print(f"Deterrent status: {status}")

        return detected_threats, threat, status, attempt_description, now < self.iot_message_until

# ==========================================
# LOAD MODELS
# ==========================================

coco_model = YOLO("yolov8n.pt")
krishi_model = YOLO("best.pt")

CUSTOM_THREATS = {
    "Crow",
    "Elephant",
    "Monkey",
    "Wildboar"
}

NON_THREATS = {
    "person",
    "dog",
    "cat",
    "horse",
    "sheep",
    "cow"
}


# ==========================================
# IOU FUNCTION
# ==========================================

def iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0

    return intersection / union


# ==========================================
# WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("KrishiNova AI started.")
print("Press Q to quit.")

alert_engine = AlertEngine(cooldown_seconds=10)

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break

    detections = []

    # ======================================
    # COCO DETECTOR
    # ======================================

    coco_result = coco_model.predict(
        frame,
        conf=0.40,
        verbose=False
    )[0]

    coco_boxes = []

    for box in coco_result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        name = coco_model.names[class_id]

        coords = box.xyxy[0].cpu().numpy().tolist()

        coco_boxes.append({
            "name": name,
            "confidence": confidence,
            "box": coords
        })

        # Generic bird detection
        if name == "bird":

            detections.append({
                "name": "Bird",
                "confidence": confidence,
                "box": coords,
                "source": "COCO"
            })

        elif name == "cow":

            detections.append({
                "name": "cow",
                "confidence": confidence,
                "box": coords,
                "source": "COCO"
            })


    # ======================================
    # KRISHINOVA CUSTOM DETECTOR
    # ======================================

    krishi_result = krishi_model.predict(
        frame,
        conf=0.50,
        verbose=False
    )[0]

    for box in krishi_result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        name = krishi_model.names[class_id]

        coords = box.xyxy[0].cpu().numpy().tolist()

        if name not in CUSTOM_THREATS:
            continue

        # Check whether custom detection overlaps
        # a normal object detected by COCO.
        suppressed = False

        for obj in coco_boxes:

            if obj["name"] in NON_THREATS:

                overlap = iou(coords, obj["box"])

                if overlap > 0.30:
                    suppressed = True
                    break

        if not suppressed:

            detections.append({
                "name": name,
                "confidence": confidence,
                "box": coords,
                "source": "KrishiNova"
            })

    detected_threats, selected_threat, alert_status, attempt_description, iot_active = alert_engine.update(detections)


    # ======================================
    # DRAW DETECTIONS
    # ======================================

    for detection in detections:

        x1, y1, x2, y2 = map(
            int,
            detection["box"]
        )

        label = (
            f"{detection['name']} "
            f"{detection['confidence']:.2f}"
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    # ======================================
    # STATUS
    # ======================================

    if detections:

        cv2.putText(
            frame,
            "CROP THREAT DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "NO CROP THREAT",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

    cv2.putText(
        frame,
        f"Detected threats: {', '.join(detected_threats) if detected_threats else 'None'}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )
    cv2.putText(
        frame,
        f"Deterrent attempt: {attempt_description}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 200, 255),
        2
    )
    cv2.putText(
        frame,
        f"Deterrent status: {alert_status}",
        (20, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 200, 255),
        2
    )

    cv2.putText(
        frame,
        "IoT Response: SIMULATED",
        (20, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    if iot_active:
        cv2.putText(
            frame,
            "Simulated IoT response activated",
            (20, 195),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )


    # ======================================
    # DISPLAY
    # ======================================

    cv2.imshow(
        "KrishiNova AI - Crop Protection",
        frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()
cv2.destroyAllWindows()

print("KrishiNova AI stopped.")