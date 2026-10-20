import cv2
import mediapipe as mp
import time

# -----------------------------
# Initialize MediaPipe
# -----------------------------
mpHands = mp.solutions.hands

hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mpDraw = mp.solutions.drawing_utils

# -----------------------------
# Open Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

previousTime = 0

while True:

    success, img = cap.read()

    if not success:
        break

    # Flip image
    img = cv2.flip(img, 1)

    # Convert BGR to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect Hands
    results = hands.process(imgRGB)

    # Draw Hand Landmarks
    if results.multi_hand_landmarks:

        for handLandmarks in results.multi_hand_landmarks:

            mpDraw.draw_landmarks(
                img,
                handLandmarks,
                mpHands.HAND_CONNECTIONS
            )

            # Print landmark positions
            for id, lm in enumerate(handLandmarks.landmark):

                h, w, c = img.shape

                cx = int(lm.x * w)
                cy = int(lm.y * h)

                # Draw a small circle
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                # Show landmark number
                cv2.putText(
                    img,
                    str(id),
                    (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1
                )

    # Calculate FPS
    currentTime = time.time()

    fps = 1 / (currentTime - previousTime)

    previousTime = currentTime

    cv2.putText(
        img,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    # Show Window
    cv2.imshow("Hand Detection", img)

    # Press Q to Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()