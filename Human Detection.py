## Human Detection
import torch
import cv2

# Load YOLOv5s model (small, fast)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=False)
model.eval()

# Define class index for 'person' in COCO
PERSON_CLASS_ID = 0

# Open webcam (change index to 0 or 1 depending on your Pi camera setup)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame)

    # Extract predictions
    detections = results.xyxy[0]  # [x1, y1, x2, y2, confidence, class]

    for *box, conf, cls in detections:
        if int(cls) == PERSON_CLASS_ID:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Person {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("YOLOv5 Person Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
## Raspberry pi with UAV
from dronekit import connect, VehicleMode
import time
import torch
import cv2

# Connect to drone
print("Connecting to drone...")
vehicle = connect("/dev/ttyACM0", wait_ready=True)

# Load YOLOv5 model
print("Loading YOLOv5 model...")
model = torch.hub.load("ultralytics/yolov5", "yolov5s", force_reload=False)
model.eval()
PERSON_CLASS_ID = 0  # COCO class ID for "person"


def arm_and_takeoff():
    print("Arming motors...")
    vehicle.mode = VehicleMode("ALT_HOLD")  # Simulated hover mode
    vehicle.armed = True

    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(1)

    print("Drone is armed. Hovering (simulated takeoff)...")
    time.sleep(3)
    print("Reached approx. 5m (simulated ALT_HOLD)")


def fly_and_detect(duration=30):
    cap = cv2.VideoCapture(0)
    detected_total = 0
    start_time = time.time()

    print(f"Starting flight for {duration} seconds with live person detection...")
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            print("Camera frame not received.")
            break

        # Run YOLOv5 model
        results = model(frame)
        detections = results.xyxy[0]

        person_count = 0
        # Draw only person class boxes
        for *xyxy, conf, cls in detections:
            if int(cls) == PERSON_CLASS_ID:
                person_count += 1
                label = f"Person {conf:.2f}"
                cv2.rectangle(
                    frame,
                    (int(xyxy[0]), int(xyxy[1])),
                    (int(xyxy[2]), int(xyxy[3])),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    label,
                    (int(xyxy[0]), int(xyxy[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        if person_count > 0:
            detected_total += person_count
            print(f"[DETECTED] {person_count} Human(s) in frame")

        # Show detection frame
        cv2.imshow("Drone Camera - Person Detection Only", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Exit requested.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(
        f"Detection complete. Total person detections during flight: {detected_total}"
    )


def land():
    print("Landing...")
    vehicle.mode = VehicleMode("LAND")
    while vehicle.armed:
        print(" Waiting for disarm...")
        time.sleep(1)
    print("Drone disarmed and landed.")


arm_and_takeoff()
fly_and_detect(duration=30)
land()
vehicle.close()
print("Mission complete.")
