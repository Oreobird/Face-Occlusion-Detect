import os
import dlib
from imutils import face_utils
import cv2
import numpy as np

class CameraTester():
    def __init__(self, net=None, input_size=96, fine_tune=False, face_landmark_path='./model/shape_predictor_68_face_landmarks.dat'):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Unable to connect to camera.")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(face_landmark_path)
        self.net = net
        self.input_size = input_size
        self.fine_tune = fine_tune
        
    def crop_face(self, shape, img, input_size):
        x = []
        y = []
        for (_x, _y) in shape:
            x.append(_x)
            y.append(_y)
        
        max_x = min(max(x), img.shape[1])
        min_x = max(min(x), 0)
        max_y = min(max(y), img.shape[0])
        min_y = max(min(y), 0)
        
        Lx = max_x - min_x
        Ly = max_y - min_y
        Lmax = int(max(Lx, Ly)) * 1.15
        delta = Lmax // 2
        
        center_x = (max(x) + min(x)) // 2
        center_y = (max(y) + min(y)) // 2
        start_x = int(center_x - delta)
        start_y = int(center_y - 0.98 * delta)
        end_x = int(center_x + delta)
        end_y = int(center_y + 1.02 * delta)

        start_y = 0 if start_y < 0 else start_y
        start_x = 0 if start_x < 0 else start_x
        end_x = img.shape[1] if end_x > img.shape[1] else end_x
        end_y = img.shape[0] if end_y > img.shape[0] else end_y
        
        crop_face = img[start_y:end_y, start_x:end_x]
        # print(crop_face.shape)
        crop_face = cv2.cvtColor(crop_face, cv2.COLOR_RGB2GRAY)
        # cv2.imshow("crop face", crop_face)
        # cv2.waitKey(0)
        crop_face = cv2.resize(crop_face, (input_size, input_size)) / 255
        channel = 3 if self.fine_tune else 1
        crop_face = np.resize(crop_face, (input_size, input_size, channel))
        return crop_face, start_y, end_y, start_x, end_x
    
    def get_area(self, shape, idx):
        #[[x, y], radius]
        left_eye = [(shape[42] + shape[45]) // 2, abs(shape[45][0] - shape[42][0])]
        right_eye = [(shape[36] + shape[39]) // 2, abs(shape[39][0] - shape[36][0])]
        nose = [shape[30], int(abs(shape[31][0] - shape[35][0]) / 1.5)]
        mouth = [(shape[48] + shape[54]) // 2, abs(shape[48][0] - shape[54][0]) // 2]
        chin = [shape[8], nose[1]]
        area = [None, right_eye, left_eye, nose, mouth, chin]
        block_area = [x for i, x in enumerate(area) if i in idx]
        return block_area
    
    def draw_occlusion_area(self, img, shape, idx):
        area = self.get_area(shape, idx)
        for k, v in enumerate(area):
            if v:
                cv2.circle(img, tuple(v[0]), v[1], (0, 255, 0))
    
    def run(self):
        frames = []
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                face_rects = self.detector(frame, 0)
                
                if len(face_rects) > 0:
                    shape = self.predictor(frame, face_rects[0])
                    shape = face_utils.shape_to_np(shape)

                    input_img, start_y, end_y, start_x, end_x = self.crop_face(shape, frame, self.input_size)

                    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), thickness=2)
                    
                    frames.append({'fod_input': input_img})
                    if len(frames) == 1:
                        pred = self.net.test_online(frames)
                  
                        # print(pred)
                        idx = [i for i, x in enumerate(pred[0]) if x > 0.8]
                        frames = []
                        # print(idx)
                        if len(idx):
                            self.draw_occlusion_area(frame, shape, idx)                    
                else:
                    print("No face detect")
                    
                cv2.imshow("frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                        
if __name__ == '__main__':
    camera_tester= CameraTester()
    camera_tester.run()
