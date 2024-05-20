from argparse import ArgumentParser
# from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.spatial import ConvexHull

from eyes import Eye_Checker

# import imageio.v2 as imageio
import cv2
# from skimage.transform import resize
# from PIL import Image

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
        eye_score = eye_checker.check_eye(image)
        if eye_score > 0.5:
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
                    print(f'EYE SCORE:  {eye_score}\t MOUTH SCORE:  {mouth_status}')
                

                elif mouth_status< 0.79 and mouth==2: #THRESHOLD
                    kp_driving = eye_checker.get_kp(image)
                    kp_driving = normalize_kp(kp_driving)
                    new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
                    norms.append(new_norm)
                    frame_nums.append(i)
                    print(f'EYE SCORE:  {eye_score}\t MOUTH SCORE:  {mouth_status}')



            else:
                kp_driving = eye_checker.get_kp(image)
                kp_driving = normalize_kp(kp_driving)
                new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
                norms.append(new_norm)
                frame_nums.append(i)
                print(f'EYE SCORE:  {eye_score}')
            
    if len(frame_nums) == 0:
        return None
    # Сортировка индексов кадров по возрастанию нормы
    sorted_frames = [x for _, x in sorted(zip(norms, frame_nums))]
    
    # Возвращаем список n лучших кадров
    return sorted_frames[:n]








def find_n_best_frames_in_source_video(source, driving, n=1, mouth=0):
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

    source_kps = []
    source_scores = []
    source_candidates = []
    for i, image in tqdm(enumerate(source)):

        eye_score = eye_checker.check_eye(image)
        if eye_score > 0.5:
            if mouth != 0:
                mouth_status = eye_checker.check_mouth(image)
                if mouth_status is None:
                    continue

                elif mouth_status>= 0.79 and mouth==1: #THRESHOLD
                    # kp_driving = eye_checker.get_kp(image)
                    # kp_driving = normalize_kp(kp_driving)
                   
                    source_candidates.append(i)
                    source_scores.append([eye_score, mouth_status])

                    kp_source = eye_checker.get_kp(image)
                    kp_source = normalize_kp(kp_source)
                    source_kps.append(kp_source)


                elif mouth_status< 0.79 and mouth==2: #THRESHOLD
                    # kp_driving = eye_checker.get_kp(image)
                    # kp_driving = normalize_kp(kp_driving)
                   
                    source_candidates.append(i)
                    source_scores.append([eye_score, mouth_status])
                    kp_source = eye_checker.get_kp(image)
                    kp_source = normalize_kp(kp_source)
                    source_kps.append(kp_source)



            else:
                # kp_driving = eye_checker.get_kp(image)
                # kp_driving = normalize_kp(kp_driving)
                
                source_candidates.append(i)
                source_scores.append([eye_score])
                kp_source = eye_checker.get_kp(image)
                kp_source = normalize_kp(kp_source)
                source_kps.append(kp_source)



    driving_kps = []
    driving_scores = []
    driving_candidates = []
    for i, image in tqdm(enumerate(driving)):

        eye_score = eye_checker.check_eye(image)
        if eye_score > 0.5:
            if mouth != 0:
                mouth_status = eye_checker.check_mouth(image)
                if mouth_status is None:
                    continue

                elif mouth_status>= 0.79 and mouth==1: #THRESHOLD
                    # kp_driving = eye_checker.get_kp(image)
                    # kp_driving = normalize_kp(kp_driving)
                   
                    driving_candidates.append(i)
                    driving_scores.append([eye_score, mouth_status])

                    kp_driving = eye_checker.get_kp(image)
                    kp_driving = normalize_kp(kp_driving)
                    driving_kps.append(kp_driving)


                elif mouth_status< 0.79 and mouth==2: #THRESHOLD
                    # kp_driving = eye_checker.get_kp(image)
                    # kp_driving = normalize_kp(kp_driving)
                   
                    driving_candidates.append(i)
                    driving_scores.append([eye_score, mouth_status])

                    kp_driving = eye_checker.get_kp(image)
                    kp_driving = normalize_kp(kp_driving)
                    driving_kps.append(kp_driving)



            else:
                # kp_driving = eye_checker.get_kp(image)
                # kp_driving = normalize_kp(kp_driving)
                
                driving_candidates.append(i)
                driving_scores.append([eye_score])

                kp_driving = eye_checker.get_kp(image)
                kp_driving = normalize_kp(kp_driving)
                driving_kps.append(kp_driving)









    norms = {}
    




    for i, dr_idx in enumerate(driving_candidates):
        for j, src_idx in enumerate(source_candidates):
            new_norm = (np.abs(source_kps[j] - driving_kps[i]) ** 2).sum()

            # if src_idx not in frame_nums:
            #     norms.append(new_norm)
            #     frame_nums.append(src_idx)
            # else:

            norms[src_idx] = new_norm




    # sorted_frames = [x for _, x in sorted(zip(norms, frame_nums))]
    
    if len(norms) == 0:
        return None
    result = sorted(norms, key=norms.get)

    
    # Возвращаем список n лучших кадров
    return result[:n]



def resize_image(image_path, target_size=(256, 256)):
# Загрузка изображения
    image = cv2.imread(image_path)
    
    # Изменение размера изображения
    # resized_image = cv2.resize(image, target_size)
    
    # Преобразование изображения в формат numpy array
    image_array = np.array(image)
    
    return image_array






def process_video(video_path, target_size=(256, 256)):
    # Загрузка видео
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Чтение и обработка каждого кадра
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Изменение размера кадра
        # resized_frame = cv2.resize(frame, target_size)
        
        # Преобразование кадра в формат numpy array и добавление в список
        frames.append(np.array(frame))
        
    cap.release()
    
    return frames




if __name__ == "__main__":
    parser = ArgumentParser()
    

    # parser.add_argument("--source_image", default='' help="path to source image")
    parser.add_argument("--source_video", help="path to source video")
    parser.add_argument("--driving_video", help="path to driving video")
    

    
    parser.add_argument("--open_mouth", dest="open_mouth", type=int, default=0, 
                        help="1 if opened mouth observing required, 2 if closed mouth observing required, else 0 - not observing (default 0)")

    parser.add_argument("--n", dest="n", type=int, default=1, 
                        help="number of best frames (default 1)")


 
    # parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
 


    opt = parser.parse_args()
    
    if opt.n <= 0:
        print('--n parametr must be at least 1')

    
    print('reading source...')
    # source_image_sk = imageio.imread(opt.source_image)
    # source_img = cv2.imread(opt.source_image)
    
    
    print('resizing...')
    # source_image_sk = np.array(resize(source_image_sk, (256, 256))[..., :3])
    # source_image_sk = (source_image_sk * 255).astype(np.uint8)
    # print('SHAPEEE source_image_sk :',source_image_sk.shape)
    # print('SHAPEEE source_image_sk[0] :', source_image_sk[0].shape)

    # source_image_cv2 = resize_image(opt.source_image)
    # print(source_image_cv2.shape)

    source_video = process_video(opt.source_video)

    # plt.imshow(source_image_cv2[0:256, 0:256, 0])
    # plt.show()

    # print(source_image_sk[0:256, 0:256, 0])
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

    # source_image = source_image_sk
    # source_image = source_image_cv2




    print('reading driving...')
    # reader = imageio.get_reader(opt.driving_video)
    # fps = reader.get_meta_data()['fps']
    # driving_video = []
    # try:
    #     for im in reader:
    #         driving_video.append(im)
    # except RuntimeError:
    #     pass
    # reader.close()

    # tmp = []
    # for i, frame in enumerate(driving_video):
    #     tmp.append(np.array(resize(frame, (256, 256))[..., :3])) 
    #     print(i)
    # # generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)
    # driving_video = tmp
   







    driving_video = process_video(opt.driving_video)


    print('params:')
    print('n: ', int(opt.n))
    print('n: ', opt.open_mouth)
    print()

    # list_of_best_frames = find_n_best_frames(source_image, driving_video, cpu=opt.cpu, n=int(opt.n), mouth = opt.open_mouth)
    list_of_best_frames = find_n_best_frames_in_source_video(source_video, driving_video, n=int(opt.n), mouth = opt.open_mouth)
    if list_of_best_frames is None:
        print('BEST FRAME NOT FOUND')
        i = 0
    else:
        i = list_of_best_frames[0]
        print('LIST OF BEST FRAMES:  ') 
        print(*list_of_best_frames)
    print ("Best frame: " + str(i))
    
    

