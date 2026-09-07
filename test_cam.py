import cv2

# Try opening the default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam. Make sure another app isn't using it.")
    exit()

print("Camera test started successfully! Press 'q' in the video window to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break
    
    # Mirror the frame
    frame = cv2.flip(frame, 1)
    
    # Overlay text
    cv2.putText(frame, "Project BMO - Camera Test", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Project BMO - Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()