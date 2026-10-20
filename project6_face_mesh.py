import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Face Mesh
mpFaceMesh = mp.solutions.face_mesh

faceMesh = mpFaceMesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mpDraw = mp.solutions.drawing_utils

drawingSpec = mpDraw.DrawingSpec(
    color=(0,255,0),
    thickness=1,
    circle_radius=1
)

cap = cv2.VideoCapture(0)

previousTime = 0

while True:

    success, img = cap.read()

    if not success:
        break

    img = cv2.flip(img,1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:

        for face in results.multi_face_landmarks:

            mpDraw.draw_landmarks(
                img,
                face,
                mpFaceMesh.FACEMESH_TESSELATION,
                drawingSpec,
                drawingSpec
            )

            h, w, c = img.shape

            # Draw landmark numbers
            for id, lm in enumerate(face.landmark):

                x = int(lm.x * w)
                y = int(lm.y * h)

                if id % 20 == 0:
                    cv2.putText(
                        img,
                        str(id),
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255,0,0),
                        1
                    )

    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(
        img,
        f"FPS: {int(fps)}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,255),
        2
    )

    cv2.imshow("Face Mesh", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()