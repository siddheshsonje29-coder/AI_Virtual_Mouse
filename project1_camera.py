import cv2

# Open the default webcam (0)
camera = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera started.")
print("Click on the camera window and press 'q' or 'Esc' to exit.")

while True:
    # Read a frame from the camera
    success, frame = camera.read()

    if not success:
        print("Failed to capture frame.")
        break

    # Display the frame
    cv2.imshow("My Camera", frame)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    # Exit if 'q' or ESC is pressed
    if key == ord('q') or key == 27:
        print("Exiting...")
        break

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()