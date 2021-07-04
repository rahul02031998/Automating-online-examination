# -*- coding: utf-8 -*-
import cv2
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks, draw_marks

def mouthCommandR():

    # face_model,landmark_model,outer_points,d_outer,inner_points,d_inner,font=getMouth()
    face_model1 = get_face_detector()
    landmark_model1 = get_landmark_model()
    outer_points1 = [[49, 59], [50, 58], [51, 57], [52, 56], [53, 55]]
    d_outer1 = [0] * 5
    inner_points1 = [[61, 67], [62, 66], [63, 65]]
    d_inner1 = [0] * 3
    font1 = cv2.FONT_HERSHEY_SIMPLEX
    from cap import openCamera, destroy, destroyAll
    cap = openCamera()


    while (True):
        ret, img = cap.read()
        rects = find_faces(img, face_model1)
        for rect in rects:
            shape = detect_marks(img, landmark_model1, rect)
            draw_marks(img, shape)
            cv2.putText(img, 'Press r to record Mouth distances', (30, 30), font1,
                        1, (255, 0, 0), 2)
            cv2.imshow("Output", img)
        if cv2.waitKey(1) & 0xFF == ord('r'):
            for i in range(100):
                for i, (p1, p2) in enumerate(outer_points1):
                    d_outer1[i] += shape[p2][1] - shape[p1][1]
                for i, (p1, p2) in enumerate(inner_points1):
                    d_inner1[i] += shape[p2][1] - shape[p1][1]
            break
        # cv2.destroyAllWindows()

    destroyAll()
    d_outer1[:] = [x / 100 for x in d_outer1]
    d_inner1[:] = [x / 100 for x in d_inner1]

    while (True):
        ret, img = cap.read()
        rects = find_faces(img, face_model1)
        for rect in rects:
            shape = detect_marks(img, landmark_model1, rect)
            cnt_outer = 0
            cnt_inner = 0
            draw_marks(img, shape[48:])
            for i, (p1, p2) in enumerate(outer_points1):
                if d_outer1[i] + 3 < shape[p2][1] - shape[p1][1]:
                    cnt_outer += 1
            for i, (p1, p2) in enumerate(inner_points1):
                if d_inner1[i] + 2 < shape[p2][1] - shape[p1][1]:
                    cnt_inner += 1
            if cnt_outer > 3 and cnt_inner > 2:
                print('Mouth open')
                cv2.putText(img, 'Mouth open', (30, 30), font1,
                            1, (255, 0, 0), 2)
            # show the output image with the face detections + facial landmarks
        cv2.imshow("Output", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    destroy()

#mouthCommandR()

