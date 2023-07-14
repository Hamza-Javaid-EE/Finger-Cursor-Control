import cv2
import mediapipe as mp
import time
import pydirectinput
import screeninfo

# Get screen resolution
screen_info = screeninfo.get_monitors()[0]
screen_width, screen_height = screen_info.width, screen_info.height

# Camera Access
cap = cv2.VideoCapture(0)

# Set processing resolution
cap.set(3, 640)
cap.set(4, 480)

# Using hand Detection Module Builtin
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

cTime = 0
pTime = 0

# Calling Camera
while True:
    success, img = cap.read()

    # Flip the image horizontally to resolve the mirror effect
    img = cv2.flip(img, 1)

    # Image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Extract the multiple Hands
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            # Access each 20 points separately
            for id, lm in enumerate(handlms.landmark):
                # This gives the each point(id) and their landmark(x,y,z) location
                # x,y,z is a ratio so we convert it into the pixel by multiplying it with width and height
                h, w, c = img.shape  # c for column
                cx, cy = int(lm.x * w), int(lm.y * h)  # Centre point

                # Scale the cursor position according to screen resolution and cursor speed
                cursor_x = int(lm.x * screen_width)
                cursor_y = int(lm.y * screen_height)

                # Control cursor movement with index finger (id=8)
                if id == 8:
                    # Increase the size of the circle for id=8
                    cv2.circle(img, (cx, cy), 20, (255, 0, 255), cv2.FILLED)
                    pydirectinput.moveTo(cursor_x, cursor_y)

                else:
                    # Make the circles smaller for other points
                    cv2.circle(img, (cx, cy), 6, (0, 255, 0), cv2.FILLED)

                # Control mouse click with middle finger (id=12) tapping or overlapping index finger (id=8)
                if id == 12:
                    finger_8 = handlms.landmark[8]  # Get the index finger landmark
                    finger_12 = handlms.landmark[12]  # Get the middle finger landmark

                    # Check if the middle finger is overlapping the index finger
                    if finger_12.y < finger_8.y:
                        # Change the color of the circle to green when id=12 overlaps id=8 for click
                        cv2.circle(img, (cx, cy), 8, (0, 255, 0), cv2.FILLED)
                        pydirectinput.click(button='left')

            # Drawing the 21 points with the MediaPipe
            mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)

    # FPS Calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
