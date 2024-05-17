# import dlib
import time
import cv2
import argparse, os, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
from .model import model_static
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from colour import Color
from .eye_logger import FaceDetectorLogger, FaceLandmarkerLogger

import imageio.v2 as imageio



class Eye_Checker:
    def __init__(self, model_weight_path):
        print('CREATING EYE_CHECKER')
        print('CREATING EYE_LOGGER')
        self.eye_logger = FaceDetectorLogger(video_mode=False)
        print(' EYE_LOGGER CREATED')
        print(' LOADING WEIGHTS')
        self.model_weight = model_weight_path

        self.model = model_static(self.model_weight)
        self.model_dict = self.model.state_dict()
        snapshot = torch.load(self.model_weight, map_location=torch.device('cpu'))
        self.model_dict.update(snapshot)
        self.model.load_state_dict(self.model_dict)
        ### UNCOMMENT -> GPU
        # model.cuda() 
        self.model.train(False)
        print('  WEIGHTS LOADED')



        self.red = Color("red")
        self.colors = list(self.red.range_to(Color("green"),10))
        self.font = ImageFont.truetype("data/arial.ttf", 40)



        self.mouth_logger = FaceLandmarkerLogger(video_mode=False)




    def bbox_jitter(self, bbox_left, bbox_top, bbox_right, bbox_bottom):
        cx = (bbox_right+bbox_left)/2.0
        cy = (bbox_bottom+bbox_top)/2.0
        scale = random.uniform(0.8, 1.2)
        bbox_right = (bbox_right-cx)*scale + cx
        bbox_left = (bbox_left-cx)*scale + cx
        bbox_top = (bbox_top-cy)*scale + cy
        bbox_bottom = (bbox_bottom-cy)*scale + cy
        return bbox_left, bbox_top, bbox_right, bbox_bottom


    def drawrect(self, drawcontext, xy, outline=None, width=0):
        (x1, y1), (x2, y2) = xy
        points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
        drawcontext.line(points, fill=outline, width=width)




    def get_kp(self, frame, frame_time=0):
        kp = self.mouth_logger.kp(frame, frame_time_nano=frame_time)
        if kp is None:
            return None
        else:
            return kp

    def check_eye(self, frame):
        # set up vis settings
        

        # set up video source
        # if video_path is None:
        #     cap = cv2.VideoCapture(0)
        #     video_path = 'live.avi'
        # else:
        #     cap = cv2.VideoCapture(video_path)

        # set up output file
        # if save_text:
        #     outtext_name = os.path.basename(video_path).replace('.avi','_output.txt')
        #     f = open(outtext_name, "w")
        # if vis:
        #     outvis_name = os.path.basename(video_path).replace('.avi','_output.avi')
        #     imwidth = int(cap.get(3)); imheight = int(cap.get(4))
        #     outvid = cv2.VideoWriter(outvis_name,cv2.VideoWriter_fourcc('M','J','P','G'), cap.get(5), (imwidth,imheight))

        # set up face detection mode
        # if face_path is None:
        #     facemode = 'DLIB'
        # else:
        #     facemode = 'GIVEN'
        #     column_names = ['frame', 'left', 'top', 'right', 'bottom']
        #     df = pd.read_csv(face_path, names=column_names, index_col=0)
        #     df['left'] -= (df['right']-df['left'])*0.2
        #     df['right'] += (df['right']-df['left'])*0.2
        #     df['top'] -= (df['bottom']-df['top'])*0.1
        #     df['bottom'] += (df['bottom']-df['top'])*0.1
        #     df['left'] = df['left'].astype('int')
        #     df['top'] = df['top'].astype('int')
        #     df['right'] = df['right'].astype('int')
        #     df['bottom'] = df['bottom'].astype('int')

        # if (cap.isOpened()== False):
        #     print("Error opening video stream or file")
        #     exit()

        # if facemode == 'DLIB':
        #     cnn_face_detector = dlib.cnn_face_detection_model_v1(CNN_FACE_MODEL)

            #### RED: here add mp detector
        # frame_cnt = 0

        # set up data transformation
        test_transforms = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        # load model weights
       

        # video reading loop
      
            # ret, frame = cap.read()
        
        height, width, channels = frame.shape
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # frame_cnt += 1
        bbox = []
        
        # dets = cnn_face_detector(frame, 1)
        try:
            index, box, conf_score = self.eye_logger.detect_and_log(frame)
        except:
            print('  FACE NOT FOUND')
            return 0
        #### RED: here add mp dets
        # index, box, conf_score = self.eye_logger.detect_and_log(frame)
        print('  FACE DETECTED')

        l = box.origin_x
        r = box.origin_x + box.width
        t = box.origin_y
        b = box.origin_y + box.height
        # expand a bit
        # l -= (r-l)*0.2
        # r += (r-l)*0.2
        # t -= (b-t)*0.2
        # b += (b-t)*0.2
        bbox.append([l,t,r,b])
        

        # frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame = Image.fromarray(frame)

        for b in bbox:
            face = frame.crop((b))
            img = test_transforms(face)
            img.unsqueeze_(0)
            # if jitter > 0:
            #     for i in range(jitter):
            #         bj_left, bj_top, bj_right, bj_bottom = bbox_jitter(b[0], b[1], b[2], b[3])
            #         bj = [bj_left, bj_top, bj_right, bj_bottom]
            #         facej = frame.crop((bj))
            #         img_jittered = test_transforms(facej)
            #         img_jittered.unsqueeze_(0)
            #         img = torch.cat([img, img_jittered])

            # forward pass
            # output = model(img.cuda())
            output = self.model(img)
            # if jitter > 0:
            #     output = torch.mean(output, 0)
            score = F.sigmoid(output).item()
            print('SCORE: ', score)

            # coloridx = min(int(round(score*10)),9)
            # draw = ImageDraw.Draw(frame)
            # self.drawrect(draw, [(b[0], b[1]), (b[2], b[3])], outline=self.colors[coloridx].hex, width=5)
            # draw.text((b[0],b[3]), str(round(score,2)), fill=(255,255,255,128), font=self.font)
            # if save_text:
            #     f.write("%d,%f\n"%(frame_cnt,score))

        # if not display_off:
            # frame = np.asarray(frame) # convert PIL image back to opencv format for faster display
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # while True:
            #     cv2.imshow('',frame)
            
            # if vis:
            #     outvid.write(frame)
                # key = cv2.waitKey(/

        # if vis:
        #     outvid.release()
        # if save_text:
        #     f.close()
        # cap.release()
        print('DONE!')
        return score


    def check_mouth(self, frame, frame_time=0):
        mouth_status = self.mouth_logger.detect_and_log(frame, frame_time_nano=frame_time)
        if mouth_status is None:
            return None
        else:
            return mouth_status

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()

    # parser.add_argument('--video', type=str, help='input video path. live cam is used when not specified')
    # parser.add_argument('--face', type=str, help='face detection file path. dlib face detector is used when not specified')
    # parser.add_argument('--model_weight', type=str, help='path to model weights file', default='data/model_weights.pkl')
    # parser.add_argument('--jitter', type=int, help='jitter bbox n times, and average results', default=0)
    # parser.add_argument('-save_vis', help='saves output as video', action='store_true')
    # parser.add_argument('-save_text', help='saves output as text', action='store_true')
    # parser.add_argument('-display_off', help='do not display frames', action='store_true')

    # args = parser.parse_args()

    # CNN_FACE_MODEL = 'data/mmod_human_face_detector.dat' # from http://dlib.net/files/mmod_human_face_detector.dat.bz2


    image_opened = r'C:\Users\msmkl\PROJECTS_PY\FaceRelator_14_05\eye-contact-cnn\noear.jpg'
    image_closed = r'C:\Users\msmkl\PROJECTS_PY\FaceRelator_14_05\eye-contact-cnn\closed_eyes.jpg'
    image_opened_m= r'C:\Users\msmkl\PROJECTS_PY\FaceRelator_14_05\eye-contact-cnn\open_m_open.jpg'
    image_cloded_m= r'C:\Users\msmkl\PROJECTS_PY\FaceRelator_14_05\eye-contact-cnn\closed_m_open.jpg'

    frame1 = imageio.imread(image_opened)
    frame2 = imageio.imread(image_closed)
    frame3 = imageio.imread(image_opened_m)
    frame4 = imageio.imread(image_cloded_m)

    print(frame1.shape)

    eye_checker = Eye_Checker(r'C:\Users\msmkl\PROJECTS_PY\FaceRelator_14_05\eye-contact-cnn\data\model_weights.pkl')
    
    # eye_checker.run(args.video, args.face, args.model_weight, args.jitter, args.save_vis, args.display_off, args.save_text)

    score1 = eye_checker.check_eye(frame1)
    mouth1 = eye_checker.check_mouth(frame1)
    print("MOUTH 1 SCORE: ",mouth1)

    score2 = eye_checker.check_eye(frame2)
    mouth2 = eye_checker.check_mouth(frame2)
    print("MOUTH 2 SCORE: ", mouth2)

    score3 = eye_checker.check_eye(frame3)
    mouth3 = eye_checker.check_mouth(frame3)
    print("MOUTH 3 SCORE: ",mouth3)

    score4= eye_checker.check_eye(frame4)
    mouth4 = eye_checker.check_mouth(frame4)
    print("MOUTH 4 SCORE: ", mouth4)