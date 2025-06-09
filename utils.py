import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils

def draw_hand_landmarks(image, hand_landmarks):
    """Draw landmarks and connections on the hand."""
    mp_drawing.draw_landmarks(
        image,
        hand_landmarks,
        mp.solutions.hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(0,128,0), thickness=2))

def put_text(img, text, position, font_scale=0.7, color=(255, 255, 255), thickness=1, bold=False):
    """Put text on the image with optional bold style."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    if bold:
        # Draw text multiple times with slight offset for bold effect
        cv2.putText(img, text, position, font, font_scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

def calculate_fps(prev_time):
    """Calculate FPS based on previous frame time."""
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    return fps, curr_time
