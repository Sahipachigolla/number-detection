import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    finger_count = 0
    h, w, _ = frame.shape

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            lm = hand_landmarks.landmark
            handedness = hand_handedness.classification[0].label

        # 🧠 Thumb logic depends on hand
        if handedness == 'Right':
            if lm[4].x > lm[3].x:
                finger_count += 1
        else:  # Left hand
            if lm[4].x < lm[3].x:
                finger_count += 1

        # 🙌 Common logic for other fingers (index, middle, ring, pinky)
        for tip_id in [8, 12, 16, 20]:
            if lm[tip_id].y < lm[tip_id - 2].y:
                finger_count += 1


            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f'Fingers: {finger_count}', (10, 70),
                cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 3)

    cv2.imshow("Finger Counter", frame)

    if cv2.waitKey(1) == 13:  # Press Enter to quit
        break

cap.release()
cv2.destroyAllWindows()
