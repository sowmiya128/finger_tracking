import cv2
import mediapipe as mp
import time
from collections import deque

# Initialize mediapipe hands and drawing utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Drawing parameters
DRAW_COLOR = (0, 0, 255)  # Red
TEXT_COLOR = (0, 0, 255)  # Red for finger name near fingertip
LABEL_COLOR = (255, 255, 255)  # White for labels (hand + gesture)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# For drawing points and lines
drawing_points = deque(maxlen=1024)
last_draw_time = 0
DRAW_TIMEOUT = 2  # seconds after which drawing disappears

# Bold text helper function
def put_bold_text(img, text, pos, font_scale=1, color=(0,0,255), thickness=2):
    # Draw black shadows for bold effect
    cv2.putText(img, text, (pos[0]-1, pos[1]-1), FONT, font_scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, (pos[0]+1, pos[1]+1), FONT, font_scale, (0,0,0), thickness+2, cv2.LINE_AA)
    # Draw main text
    cv2.putText(img, text, pos, FONT, font_scale, color, thickness, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(0)
    prev_time = 0

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:

        global last_draw_time

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Mirror image
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            gesture_texts = []

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(0,128,0), thickness=2))

                    # Label (Right or Left)
                    hand_label = handedness.classification[0].label  # 'Right' or 'Left'

                    lm_list = []
                    for lm in hand_landmarks.landmark:
                        lm_list.append(lm)

                    # Finger status calculation
                    fingers = []

                    # Thumb
                    if hand_label == "Right":
                        fingers.append(lm_list[mp_hands.HandLandmark.THUMB_TIP].x < lm_list[mp_hands.HandLandmark.THUMB_IP].x)
                    else:
                        fingers.append(lm_list[mp_hands.HandLandmark.THUMB_TIP].x > lm_list[mp_hands.HandLandmark.THUMB_IP].x)

                    # Other four fingers
                    for tip_id in [mp_hands.HandLandmark.INDEX_FINGER_TIP,
                                   mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                                   mp_hands.HandLandmark.RING_FINGER_TIP,
                                   mp_hands.HandLandmark.PINKY_TIP]:
                        fingers.append(lm_list[tip_id].y < lm_list[tip_id - 2].y)

                    total_fingers = fingers.count(True)

                    # Gesture detection logic
                    if total_fingers == 5:
                        gesture_name = "Stop"
                    elif fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
                        gesture_name = "Pointer"
                    elif total_fingers == 1:
                        finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                        for i, val in enumerate(fingers):
                            if val:
                                gesture_name = finger_names[i]
                                break
                    else:
                        gesture_name = "No Gesture"

                    # Draw hand label + gesture near wrist (use wrist landmark)
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    wrist_pos = (int(wrist.x * w), int(wrist.y * h) + 30)

                    cv2.putText(frame, f"{hand_label}: {gesture_name}",
                                wrist_pos, FONT, 0.7, LABEL_COLOR, 2, cv2.LINE_AA)

                    # Drawing logic: when only index finger is up (pointer)
                    if gesture_name == "Pointer":
                        # Get fingertip position (index finger tip)
                        index_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        x, y = int(index_fingertip.x * w), int(index_fingertip.y * h)

                        drawing_points.append((x, y))
                        last_draw_time = time.time()

                        # Draw small circle at fingertip
                        cv2.circle(frame, (x, y), 8, DRAW_COLOR, -1)

                        # Draw "Pointer" text near fingertip in bold, red, size a bit bigger
                        put_bold_text(frame, "Pointer", (x + 10, y - 10), font_scale=1, color=TEXT_COLOR, thickness=2)

            # Draw on the frame all collected points as a line
            if drawing_points:
                for i in range(1, len(drawing_points)):
                    if drawing_points[i - 1] is None or drawing_points[i] is None:
                        continue
                    cv2.line(frame, drawing_points[i - 1], drawing_points[i], DRAW_COLOR, 5)

                # Clear drawing if timeout exceeded
                if time.time() - last_draw_time > DRAW_TIMEOUT:
                    drawing_points.clear()

            # Show FPS
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), FONT, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Hand Gesture Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
