from argparse import ArgumentParser
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.spatial import ConvexHull

from eyes import Eye_Checker

import imageio.v2 as imageio
import cv2
from skimage.transform import resize
from PIL import Image

def find_n_best_frames(source, driving, n=1, cpu=False, mouth=0):
    # import face_alignment

    print('CREATING eye_checker entity...')
    eye_checker = Eye_Checker(r'eyes\data\model_weights.pkl')
    print('=== eye_checker created ===')
    
    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
    #                                   device='cpu' if cpu else 'cuda')

    kp_source = eye_checker.get_kp(source)
    # kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norms = []
    frame_nums = []
    for i, image in tqdm(enumerate(driving)):

        if eye_checker.check_eye(image) > 0.5:
            if mouth != 0:
                mouth_status = eye_checker.check_mouth(image)
                if mouth_status is None:
                    continue

                elif mouth_status>= 0.79 and mouth==1: #THRESHOLD
                    kp_driving = eye_checker.get_kp(image)
                    kp_driving = normalize_kp(kp_driving)
                    new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
                    norms.append(new_norm)
                    frame_nums.append(i)
                

                elif mouth_status< 0.79 and mouth==2: #THRESHOLD
                    kp_driving = eye_checker.get_kp(image)
                    kp_driving = normalize_kp(kp_driving)
                    new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
                    norms.append(new_norm)
                    frame_nums.append(i)



            else:
                kp_driving = eye_checker.get_kp(image)
                kp_driving = normalize_kp(kp_driving)
                new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
                norms.append(new_norm)
                frame_nums.append(i)
            
    if len(frame_nums) == 0:
        return None
    # Сортировка индексов кадров по возрастанию нормы
    sorted_frames = [x for _, x in sorted(zip(norms, frame_nums))]
    
    # Возвращаем список n лучших кадров
    return sorted_frames[:n]

if __name__ == "__main__":
    parser = ArgumentParser()
    

    parser.add_argument("--source_image", default='sup-mat/source.png', help="path to source image")
    parser.add_argument("--driving_video", default='sup-mat/source.png', help="path to driving video")
    
    
    
    parser.add_argument("--open_mouth", dest="open_mouth", type=int, default=0, 
                        help="1 if opened mouth observing required, 2 if closed mouth observing required, else 0 - not observing (default 0)")

    parser.add_argument("--n", dest="n", type=int, default=1, 
                        help="number of best frames (default 1)")


 
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
 


    opt = parser.parse_args()
    
    if opt.n <= 0:
        print('--n parametr must be at least 1')

    
    print('reading image...')
    source_image_sk = imageio.imread(opt.source_image)
    source_img = cv2.imread(opt.source_image)
    
    
    print('resizing...')
    source_image_sk = np.array(resize(source_image_sk, (256, 256))[..., :3])
    source_image_sk = (source_image_sk * 255).astype(np.uint8)
    print('SHAPEEE source_image_sk :',source_image_sk.shape)
    print('SHAPEEE source_image_sk[0] :', source_image_sk[0].shape)

    plt.imshow(source_image_sk[0:256, 0:256, 0])
    plt.show()

    print(source_image_sk[0:256, 0:256, 0])
    # resized_frame = np.uint16(cv2.resize(source_img, (256, 256)))
    # print(source_img.shape)
    # print(source_image.shape)
    # source_image = source_img
    # source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    
    # cv2.imshow('source', source_image)



    # base_width = 256
    # img = Image.open(opt.source_image)
    # wpercent = (base_width / float(img.size[0]))
    # hsize = int((float(img.size[1]) * float(wpercent)))
    # img = img.resize((256, 256), Image.Resampling.LANCZOS)
    # img.save('somepic.png')

    # print(np.array(img).shape)
    source_image = source_image_sk


    print('reading video...')
    reader = imageio.get_reader(opt.driving_video)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    tmp = []
    for i, frame in enumerate(driving_video):
        tmp.append(np.array(resize(frame, (256, 256))[..., :3])) 
        print(i)
    # generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)
    driving_video = tmp
   
    print('params:')
    print('n: ', int(opt.n))
    print('n: ', opt.open_mouth)
    print()

    list_of_best_frames = find_n_best_frames(source_image, driving_video, cpu=opt.cpu, n=int(opt.n), mouth = opt.open_mouth)
    if list_of_best_frames is None:
        print('BEST FRAME NOT FOUND')
        i = 0
    else:
        i = list_of_best_frames[-1]
        print('LIST OF BEST FRAMES:  ') 
        print(*list_of_best_frames)
    print ("Best frame: " + str(i))
    
    

