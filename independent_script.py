from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
from scipy.spatial import ConvexHull

from eyes import Eye_Checker

import imageio.v2 as imageio

from skimage.transform import resize

def find_n_best_frames(source, driving, n=1, cpu=False, mouth=0):
    # import face_alignment


    eye_checker = Eye_Checker(r'eyes\data\model_weights.pkl')

    
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

    

    source_image = imageio.imread(opt.source_image)
    reader = imageio.get_reader(opt.driving_video)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    source_image = resize(source_image, (256, 256))[..., :3]
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
    # generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)

   

    list_of_best_frames = find_n_best_frames(source_image, driving_video, cpu=opt.cpu, n=int(opt.n), mouth = opt.open_mouth)
    if list_of_best_frames is None:
        print('BEST FRAME NOT FOUND')
        i = 0
    else:
        i = list_of_best_frames[-1]
        print('LIST OF BEST FRAMES:  ') 
        print(*list_of_best_frames)
    print ("Best frame: " + str(i))
    
    

