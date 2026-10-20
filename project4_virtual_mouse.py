import cv2
import mediapipe as mp
import pyautogui

# Camera size
camWidth = 640
camHeight = 480

# Screen size
screenWidth, screenHeight = pyautogui.size()

# MediaPipe
mpHands = mp.solutions.hands

hands = mpHands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

cap.set(3, camWidth)
cap.set(4, camHeight)

while True:

    success, img = cap.read()

    if not success:
        break

    img = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        mpDraw.draw_landmarks(
            img,
            hand,
            mpHands.HAND_CONNECTIONS
        )

        h, w, c = img.shape

        # Index finger tip
        lm = hand.landmark[8]

        x = int(lm.x * w)
        y = int(lm.y * h)

        cv2.circle(img, (x, y), 12, (0, 255, 0), cv2.FILLED)

        # Convert camera coordinates to screen coordinates
        screenX = screenWidth / camWidth * x
        screenY = screenHeight / camHeight * y

        pyautogui.moveTo(screenX, screenY)

    cv2.imshow("AI Virtual Mouse", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()