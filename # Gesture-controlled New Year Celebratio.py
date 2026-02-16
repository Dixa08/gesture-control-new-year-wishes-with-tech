import cv2
import mediapipe as mp
import math

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ---------- Drawing Functions ----------

def draw_balloon(img, x, y, color):
    cv2.circle(img, (x, y), 25, color, -1)          # balloon
    cv2.line(img, (x, y + 25), (x, y + 70), (255, 255, 255), 2)  # string

def draw_flower(img, x, y):
    for angle in range(0, 360, 60):
        rad = math.radians(angle)
        px = int(x + 20 * math.cos(rad))
        py = int(y + 20 * math.sin(rad))
        cv2.circle(img, (px, py), 10, (0, 0, 255), -1)  # petals
    cv2.circle(img, (x, y), 10, (0, 255, 255), -1)      # center

# ---------- Main Loop ----------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                # Count raised fingers
                finger_tips = [8, 12, 16, 20]
                count = 0
                for tip in finger_tips:
                    if handLms.landmark[tip].y < handLms.landmark[tip - 2].y:
                        count += 1

                # Gesture trigger (3 fingers)
                if count == 3:
                    # ---- Message ----
                    cv2.putText(frame, "HAPPY NEW YEAR 2026",
                                (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, (0, 255, 0), 3)

                    cv2.putText(frame, "May God bless you with",
                                (50, 120), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 255), 2)

                    cv2.putText(frame, "Happiness, Health & Success",
                                (50, 155), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 255), 2)

                    # ---- Balloons ----
                    draw_balloon(frame, 50, 250, (255, 0, 0))
                    draw_balloon(frame, 120, 230, (0, 255, 0))
                    draw_balloon(frame, 190, 250, (0, 0, 255))

                    # ---- Flowers ----
                    draw_flower(frame, 450, 260)
                    draw_flower(frame, 520, 260)
                    draw_flower(frame, 485, 310)

        cv2.imshow("New Year Gesture Celebration", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released safely")
