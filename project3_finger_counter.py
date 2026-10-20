import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

tipIds = [4, 8, 12, 16, 20]

while True:

    success, img = cap.read()

    if not success:
        break

    img = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    lmList = []

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        mpDraw.draw_landmarks(
            img,
            hand,
            mpHands.HAND_CONNECTIONS
        )

        h, w, c = img.shape

        for id, lm in enumerate(hand.landmark):

            cx = int(lm.x * w)
            cy = int(lm.y * h)

            lmList.append((cx, cy))

            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

    if len(lmList) != 0:

        fingers = []

        # Thumb
        if lmList[4][0] > lmList[3][0]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Four Fingers
        for tip in [8, 12, 16, 20]:

            if lmList[tip][1] < lmList[tip - 2][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)

        cv2.putText(
            img,
            f"Fingers: {totalFingers}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )

    cv2.imshow("Finger Counter", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()