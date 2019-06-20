import numpy as np
import cv2


def main():
    # These are found in the opencv source at opencv/data/haarcascades/
    face_cascade = cv2.CascadeClassifier("opencv/data/haarcascades/haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("opencv/data/haarcascades/haarcascade_eye.xml")

    font = cv2.FONT_HERSHEY_SIMPLEX

    cap = cv2.VideoCapture(0)
    # Camera resolution -
    # cap.set(3, 640) # Width
    # cap.set(4, 480) # Height

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here. Convert to gray image for faster video processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50)
        )

        # If at least 1 face detected
        if len(faces) >= 0:

            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(
                    img=frame,
                    text="FACE",
                    org=(x, y - 5),
                    fontFace=font,
                    fontScale=0.5,
                    color=(255, 0, 0),
                )

                # Eye detection
                # Use the location of the face, since eyes are on the face
                roi_gray = gray[y : y + h, x : x + w]
                roi_color = frame[y : y + h, x : x + w]

                eyes = eye_cascade.detectMultiScale(roi_gray)

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(
                        roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2
                    )
                    cv2.putText(
                        img=roi_color,
                        text="EYE",
                        org=(ex, ey - 5),
                        fontFace=font,
                        fontScale=0.5,
                        color=(0, 255, 0),
                    )

        # Dislplay the resutling frame
        cv2.imshow("frame", frame)

        # Wait for q to close the app
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
