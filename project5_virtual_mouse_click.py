import cv2
import mediapipe as mp
import pyautogui
import math

# Camera Resolution
camWidth = 640
camHeight = 480

# Frame Reduction
frameR = 100

# Smoothening
smoothening = 7

# Previous Mouse Location
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# Screen Size
screenWidth, screenHeight = pyautogui.size()

# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)

# MediaPipe
mpHands = mp.solutions.hands

hands = mpHands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mpDraw = mp.solutions.drawing_utils

pyautogui.FAILSAFE = False

while True:

    success, img = cap.read()

    if not success:
        break

    img = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    # Draw Control Area
    cv2.rectangle(
        img,
        (frameR, frameR),
        (camWidth-frameR, camHeight-frameR),
        (255,0,255),
        2
    )

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        mpDraw.draw_landmarks(
            img,
            hand,
            mpHands.HAND_CONNECTIONS
        )

        h, w, c = img.shape

        lmList = []

        for lm in hand.landmark:

            cx = int(lm.x * w)
            cy = int(lm.y * h)

            lmList.append((cx, cy))

        # Index Finger Tip
        x1, y1 = lmList[8]

        # Thumb Tip
        x2, y2 = lmList[4]

        # Convert Coordinates
        mouseX = (x1-frameR) * screenWidth / (camWidth-2*frameR)
        mouseY = (y1-frameR) * screenHeight / (camHeight-2*frameR)

        # Smooth Movement
        clocX = plocX + (mouseX-plocX)/smoothening
        clocY = plocY + (mouseY-plocY)/smoothening

        pyautogui.moveTo(clocX, clocY)

        plocX, plocY = clocX, clocY

        # Draw Index Finger
        cv2.circle(img,(x1,y1),10,(0,255,0),cv2.FILLED)

        # Draw Thumb
        cv2.circle(img,(x2,y2),10,(255,0,0),cv2.FILLED)

        # Distance Between Fingers
        distance = math.hypot(x2-x1,y2-y1)

        cv2.line(img,(x1,y1),(x2,y2),(0,255,255),2)

        cv2.putText(
            img,
            f"Distance: {int(distance)}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        # Click Gesture
        if distance < 35:

            cv2.circle(img,((x1+x2)//2,(y1+y2)//2),12,(0,0,255),cv2.FILLED)

            pyautogui.click()

            cv2.putText(
                img,
                "CLICK",
                (20,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                3
            )

    cv2.imshow("AI Virtual Mouse", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()