import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import json
import time


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

PERSON_HEIGHT_M = 1.7
FRAME_RATE = 30
y_positions = []
times = []
success_count = 0
error_count = 0

pixel_to_meter = None
standing_pixel_height = None

print("Stand still for 2 seconds to calibrate height, then press 'j' to jump, 'q' to quit.")

start_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't read video frame!")
        error_count += 1
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    try:
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            hip_y = results.pose_landmarks.landmark[23].y * frame.shape[0]
            y_positions.append(hip_y)
            times.append(time.time() - start_time)


            if standing_pixel_height is None and times[-1] < 2:
                shoulder_y = results.pose_landmarks.landmark[11].y * frame.shape[0]
                ankle_y = results.pose_landmarks.landmark[27].y * frame.shape[0]
                standing_pixel_height = ankle_y - shoulder_y
                pixel_to_meter = PERSON_HEIGHT_M / standing_pixel_height
                print(f"Calibrated! Pixel-to-meter ratio: {pixel_to_meter:.6f} m/pixel")

    except Exception as e:
        print(f"Error processing frame: {e}")
        error_count += 1
        continue


    cv2.putText(frame, "Press 'j' to jump, 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Jump Height Estimator", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('j'):
        if len(y_positions) > 10:
            y_meters = [(max(y_positions) - y) * pixel_to_meter for y in y_positions]
            jump_height = max(y_meters) - min(y_meters)
            print(f"Jump Height: {jump_height * 100:.2f} cm")
            success_count += 1

        
            plt.plot(times, y_meters)
            plt.xlabel("Time (s)")
            plt.ylabel("Hip Height (m)")
            plt.title("Vertical Movement of Hip During Jump")
            plt.savefig("jump_movement.png")
            plt.close()

            chart_config = {
                "type": "bar",
                "data": {
                    "labels": ["Successful Jumps", "Errors"],
                    "datasets": [{
                        "label": "Count",
                        "data": [success_count, error_count],
                        "backgroundColor": ["#4CAF50", "#F44336"],
                        "borderColor": ["#388E3C", "#D32F2F"],
                        "borderWidth": 1
                    }]
                },
                "options": {
                    "scales": {
                        "y": {
                            "beginAtZero": True,
                            "title": {"display": True, "text": "Count"}
                        },
                        "x": {
                            "title": {"display": True, "text": "Outcome"}
                        }
                    },
                    "plugins": {
                        "legend": {"display": True},
                        "title": {"display": True, "text": "Jump Detection Outcomes"}
                    }
                }
            }
            with open("jump_outcome.json", "w") as f:
                json.dump(chart_config, f)
            print("Chart saved as jump_outcome.json")
        else:
            print("Not enough frames to calculate jump height!")
            error_count += 1


    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
print("Program ended.")