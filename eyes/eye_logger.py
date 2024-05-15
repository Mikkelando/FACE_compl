import argparse
import csv
import itertools
import logging
import math
import os
from pathlib import Path


import cv2
import mediapipe as mp
import numpy as np
import numpy.typing as npt
import requests

import tqdm
from mediapipe.tasks.python import vision

from scipy.spatial import distance as dist

MODEL_DIR = Path(os.path.dirname(__file__))

BLENDSHAPES_CATEGORIES = {
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
}


class FaceDetectorLogger:
    """
    Logger for the MediaPipe Face Detection solution.
    https://developers.google.com/mediapipe/solutions/vision/face_detector
    """

    MODEL_PATH = (MODEL_DIR / "blaze_face_short_range.tflite").resolve()
    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/"
        "blaze_face_short_range.tflite"
    )

    META = []

    def __init__(self, video_mode = False):
        self._video_mode = video_mode

        # download model if necessary
        if not self.MODEL_PATH.exists():
            download_file(self.MODEL_URL, self.MODEL_PATH)

        self._base_options = mp.tasks.BaseOptions(
            model_asset_path=str(self.MODEL_PATH),
        )
        self._options = vision.FaceDetectorOptions(
            base_options=self._base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO if self._video_mode else mp.tasks.vision.RunningMode.IMAGE,
        )
        self._detector = vision.FaceDetector.create_from_options(self._options)

        # With this annotation, the viewer will connect the keypoints with some lines to improve visibility.
        
    def detect_and_log(self, image, frame_time_nano=0, frame_count=0) :
        # image = cv2.imread(image)
        height, width, _ = image.shape
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        detection_result = (
            self._detector.detect_for_video(image, int(frame_time_nano / 1e6))
            if self._video_mode
            else self._detector.detect(image)
        )
        
        for i, detection in enumerate(detection_result.detections):

            bbox = detection.bounding_box
            index, score = detection.categories[0].index, detection.categories[0].score

            return index, bbox, score
        
        return None

           
            
           




class FaceLandmarkerLogger:
    """
    Logger for the MediaPipe Face Landmark Detection solution.
    https://developers.google.com/mediapipe/solutions/vision/face_landmarker
    """

    MODEL_PATH  = (MODEL_DIR / "face_landmarker.task").resolve()
    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/"
        "face_landmarker.task"
    )

    


    landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
                  296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
                  380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]
    

    def __init__(self, video_mode, num_faces = 1):
        self._video_mode = video_mode

        # download model if necessary
        if not self.MODEL_PATH.exists():
            download_file(self.MODEL_URL, self.MODEL_PATH)

        self._base_options = mp.tasks.BaseOptions(
            model_asset_path=str(self.MODEL_PATH),
        )
        self._options = vision.FaceLandmarkerOptions(
            base_options=self._base_options,
            output_face_blendshapes=True,
            num_faces=num_faces,
            running_mode=mp.tasks.vision.RunningMode.VIDEO if self._video_mode else mp.tasks.vision.RunningMode.IMAGE,
        )
        self._detector = vision.FaceLandmarker.create_from_options(self._options)

        classes = [
            mp.solutions.face_mesh.FACEMESH_LIPS,
            mp.solutions.face_mesh.FACEMESH_LEFT_EYE,
            mp.solutions.face_mesh.FACEMESH_LEFT_IRIS,
            mp.solutions.face_mesh.FACEMESH_LEFT_EYEBROW,
            mp.solutions.face_mesh.FACEMESH_RIGHT_EYE,
            mp.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW,
            mp.solutions.face_mesh.FACEMESH_RIGHT_IRIS,
            mp.solutions.face_mesh.FACEMESH_FACE_OVAL,
            mp.solutions.face_mesh.FACEMESH_NOSE,
        ]

        self._class_ids = [0] * mp.solutions.face_mesh.FACEMESH_NUM_LANDMARKS_WITH_IRISES
        class_descriptions = []
        for i, klass in enumerate(classes):
            # MediaPipe only provides connections for class, not actual class per keypoint. So we have to extract the
            # classes from the connections.
            ids = set()
            for connection in klass:
                ids.add(connection[0])
                ids.add(connection[1])

            for id_ in ids:
                self._class_ids[id_] = i

            

        

    def detect_and_log(self, image, frame_time_nano=0, frame_count=0) :
        height, width, _ = image.shape
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        detection_result = (
            self._detector.detect_for_video(image, int(frame_time_nano / 1e6))
            if self._video_mode
            else self._detector.detect(image)
        )

        # def is_empty(i):  # type: ignore[no-untyped-def]
        #     try:
        #         next(i)
        #         return False
        #     except StopIteration:
        #         return True

        

        for i, (landmark, blendshapes) in enumerate(
                zip(detection_result.face_landmarks, detection_result.face_blendshapes)
        ):
            if len(landmark) == 0 or len(blendshapes) == 0:
                continue

            # MediaPipe's keypoints are normalized to [0, 1], so we need to scale them to get pixel coordinates.
            pts = [(math.floor(lm.x * width), math.floor(lm.y * height)) for lm in landmark]
            keypoint_ids = list(range(len(landmark)))
            # index, score = detection_result.detections[0].categories[0].index, detection_result.detections[0].categories[0].score


     
            xs = [
                pts[i][0] for i in self.landmark_points_68
            ]
            ys = [
                pts[i][1] for i in self.landmark_points_68
            ]
            mouth_points = np.array([(xs[i], ys[i]) for i in range(49, 68)])
            

            mar = self.mouth_aspect_ratio(mouth_points)
            return mar

        return None


            
    def mouth_aspect_ratio(self, mouth):
        # compute the euclidean distances between the two sets of
        # vertical mouth landmarks (x, y)-coordinates
        A = dist.euclidean(mouth[2], mouth[10]) # 51, 59
        B = dist.euclidean(mouth[4], mouth[8]) # 53, 57

        # compute the euclidean distance between the horizontal
        # mouth landmark (x, y)-coordinates
        C = dist.euclidean(mouth[0], mouth[6]) # 49, 55

        # compute the mouth aspect ratio
        mar = (A + B) / (2.0 * C)

        # return the mouth aspect ratio
        return mar
            

            

def download_file(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Downloading %s to %s", url, path)
    response = requests.get(url, stream=True)
    with tqdm.tqdm.wrapattr(
            open(path, "wb"),
            "write",
            miniters=1,
            total=int(response.headers.get("content-length", 0)),
            desc=f"Downloading {path.name}",
    ) as f:
        for chunk in response.iter_content(chunk_size=4096):
            f.write(chunk)


# def run_from_video_capture(frame, num_faces) :
#     # cap = cv2.VideoCapture(vid)

#     # fps = cap.get(cv2.CAP_PROP_FPS)

#     detector = FaceDetectorLogger(video_mode=True)
#     # landmarker = FaceLandmarkerLogger(video_mode=True, num_faces=num_faces)

#     # print("Capturing video stream. Press ctrl-c to stop.")
#     try:
#         # it = itertools.count()

#         # for frame_idx in tqdm.tqdm(it, desc="Processing frames"):
#             # ret, frame = cap.read()
#             # if not ret:
#             #     break
#             # if np.all(frame == 0):
#             #     continue
#             # frame_time_nano = int(cap.get(cv2.CAP_PROP_POS_MSEC) * 1e6)
#             # if frame_time_nano == 0:
#             #     frame_time_nano = int(frame_idx * 1000 / fps * 1e6)

#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         detector.detect_and_log(frame, 0, 0)
#             # landmarker.detect_and_log(frame, frame_time_nano, frame_idx)
           

#     except KeyboardInterrupt:
#         pass

#     # When everything done, release the capture
#     # cap.release()
#     cv2.destroyAllWindows()


#     return detector.META, landmarker.TIMELINE


def create_csv(timeline, output_dir, name):

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(name))[0] + ".csv"

    # Создание пути к CSV-файлу
    output_file = os.path.join(output_dir, filename)

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Добавление названий колонок
        column_names = ["frame", "face_id", "timestamp", "confidence", "success"]
        xs = [f"x_{i}" for i in range(68)]
        ys = [f"y_{i}" for i in range(68)]
        column_names.extend(xs)
        column_names.extend(ys)
        
        writer.writerow(column_names)

        writer.writerows(timeline)

        print(f'CSV file created: {output_file}')

def main():
    # image = r'C:\Users\msmkl\PROJECTS_PY\FaceRelator_14_05\eye-contact-cnn\noear.jpg'
    image = r'C:\Users\msmkl\PROJECTS_PY\FaceRelator_14_05\eye-contact-cnn\closed_eyes.jpg'
    image = cv2.imread(image)
    logger = FaceDetectorLogger(video_mode=False)
    index, bbox, score = logger.detect_and_log(image)
    print(index, bbox, score)
    print(bbox.origin_y)
    # parser = argparse.ArgumentParser(description='Facial Landmarks Detection and CSV Creation')
    # parser.add_argument('video_dir', type=str, help='Path to the input video dir')
    # parser.add_argument('--output_dir', type=str, default='./csv', help='Directory to save the output CSV files')

    # args = parser.parse_args()

    # video_dir = args.video_dir
    # output_dir = args.output_dir

    # if not os.path.isdir(video_dir):
    #     print(f"Error: {video_dir} is not a valid directory.")
    #     return

    # detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


    # video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

    # for video in video_files:
    # print(f'STARTED DETECTING LANDMARKS FOR: {video}')

    # meta, timeline = run_from_video_capture(os.path.join(video_dir, video), 1)

    
    # print()
    # print('META')
    # print(meta[:5])
    # print()
    # print('timeline')
    # print(timeline[:5])


    # TIMELINE = [meta[i] + timeline[i] for i in range(len(timeline))]
    # print()
    # print()
    # print(TIMELINE[:5])
    # create_csv(TIMELINE, output_dir, video)
    

if __name__ == '__main__':
    main()  