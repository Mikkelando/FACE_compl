from argparse import ArgumentParser
import os
import imageio
import numpy as np
from tqdm import tqdm
from scipy.spatial import ConvexHull
from eyes import Eye_Checker
import cv2
from moviepy.editor import VideoFileClip
import ffmpeg


def find_n_best_frames(source, driving, n=1, cpu=False, mouth=0):
    # import face_alignment

    print('CREATING eye_checker entity...')
    eye_checker = Eye_Checker('eyes/data/model_weights.pkl')
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
    eye_checker = Eye_Checker('eyes/data/model_weights.pkl')
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

    source_kps_bad = []
    source_scores_bad = []
    source_candidates_bad = []   
    for i, image in tqdm(enumerate(source)):

        eye_score = eye_checker.check_eye(image)
        mouth_status = eye_checker.check_mouth(image)
        
        
        kp_source_bad = eye_checker.get_kp(image)
        if kp_source_bad is None:
            continue
        source_candidates_bad.append(i)
        source_scores_bad.append([eye_score])
        kp_source_bad = normalize_kp(kp_source_bad)
        source_kps_bad.append(kp_source_bad)

        if eye_score > 0.5:
            if mouth != 0:
                
                if mouth_status is None:
                    continue

                elif mouth_status>= 0.79 and mouth==1: #THRESHOLD
                    # kp_driving = eye_checker.get_kp(image)
                    # kp_driving = normalize_kp(kp_driving)
                   
                    

                    kp_source = eye_checker.get_kp(image)
                    if kp_source is None:
                        continue
                    kp_source = normalize_kp(kp_source)
                    source_kps.append(kp_source)
                    source_candidates.append(i)
                    source_scores.append([eye_score, mouth_status])


                elif mouth_status< 0.79 and mouth==2: #THRESHOLD
                    # kp_driving = eye_checker.get_kp(image)
                    # kp_driving = normalize_kp(kp_driving)
                   
                    kp_source = eye_checker.get_kp(image)
                    if kp_source is None:
                        continue
                    kp_source = normalize_kp(kp_source)
                    source_kps.append(kp_source)
                    source_candidates.append(i)
                    source_scores.append([eye_score, mouth_status])

                    
        


            else:
                # kp_driving = eye_checker.get_kp(image)
                # kp_driving = normalize_kp(kp_driving)
                
                kp_source = eye_checker.get_kp(image)
                if kp_source is None:
                    continue
                kp_source = normalize_kp(kp_source)
                source_kps.append(kp_source)
                source_candidates.append(i)
                source_scores.append([eye_score, mouth_status])

                
       
            

    print('LEN OF SOURCE CANDIDATES : ', len(source_candidates))

    # print('CREATING eye_checker_2 entity...')
    # eye_checker_2 = Eye_Checker('eyes/data/model_weights.pkl')
    # print('=== eye_checker_2 created ===')

    driving_kps = []
    driving_scores = []
    driving_candidates = []

    driving_kps_bad = []
    driving_scores_bad = []
    driving_candidates_bad = []

    

    print('TOTAL LEN OF DRIVING: ', len(driving))
    for i, image in tqdm(enumerate(driving)):

        eye_score = eye_checker.check_eye(image)
        mouth_status = eye_checker.check_mouth(image)

        
        kp_driving_bad = eye_checker.get_kp(image)
        if kp_driving_bad is None:
            continue
        kp_driving_bad = normalize_kp(kp_driving_bad)
        driving_kps_bad.append(kp_driving_bad)
        driving_candidates_bad.append(i)
        driving_scores_bad.append([eye_score])
        # print(f'eye score {i}/{len(driving)} . . . {eye_score} ')
        if eye_score > 0.5:
            if mouth != 0:
                
                if mouth_status is None:
                    continue

                elif mouth_status>= 0.79 and mouth==1: #THRESHOLD
                    # kp_driving = eye_checker.get_kp(image)
                    # kp_driving = normalize_kp(kp_driving)
                   
                    

                    kp_driving = eye_checker.get_kp(image)
                    if kp_driving is None:
                        continue
                    kp_driving = normalize_kp(kp_driving)
                    driving_kps.append(kp_driving)
                    driving_candidates.append(i)
                    driving_scores.append([eye_score, mouth_status])


                elif mouth_status< 0.79 and mouth==2: #THRESHOLD
                    # kp_driving = eye_checker.get_kp(image)
                    # kp_driving = normalize_kp(kp_driving)
                   
                    kp_driving = eye_checker.get_kp(image)
                    if kp_driving is None:
                        continue
                    kp_driving = normalize_kp(kp_driving)
                    driving_kps.append(kp_driving)
                    driving_candidates.append(i)
                    driving_scores.append([eye_score, mouth_status])



            else:
                # kp_driving = eye_checker.get_kp(image)
                # kp_driving = normalize_kp(kp_driving)
                
                kp_driving = eye_checker.get_kp(image)
                if kp_driving is None:
                    continue
                kp_driving = normalize_kp(kp_driving)
                driving_kps.append(kp_driving)
                driving_candidates.append(i)
                driving_scores.append([eye_score, mouth_status])



    print('LEN OF DRIVING CANDIDATES : ', len(driving_candidates))





    norms = {}
    


    if len(driving_candidates) > 0 and len(source_candidates) > 0:

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
    

    else:
        for i, dr_idx in enumerate(driving_candidates_bad):
            for j, src_idx in enumerate(source_candidates_bad):
                new_norm = (np.abs(source_kps_bad[j] - driving_kps_bad[i]) ** 2).sum()

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



#### CV2




def process_video(video_path, target_size=(256, 256)):
    # Загрузка видео
    print("file exists?", os.path.exists(video_path))
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    if not cap.isOpened():
        print(f"{'ERROR '*10} : {video_path}")
        return frames
    # Чтение и обработка каждого кадра
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Изменение размера кадра
        # resized_frame = cv2.resize(frame, target_size)
        
        # Преобразование кадра в формат numpy array и добавление в список
        frames.append(np.array(frame))
        
    cap.release()
    cv2.destroyAllWindows()
    print(f'{os.path.basename(video_path)} . . . done!')
    return frames



#### imageio
'''
# def process_video(video_path, target_size=(256, 256)):
#     # Загрузка видео
#     try:
#         reader = imageio.get_reader(video_path)
#     except FileNotFoundError:
#         print(f"{'ERROR '*10} : {video_path}")
#         return []

#     frames = []
#     for frame in reader:
#         # Изменение размера кадра
#         # resized_frame = cv2.resize(frame, target_size)
        
#         # Преобразование кадра в формат numpy array и добавление в список
#         frames.append(np.array(frame))
        
#     print(f'{os.path.basename(video_path)} . . . done!')
#     return frames


### MOVIEPY
# def process_video(video_path, target_size=(256, 256)):
#     # Загрузка видео
#     try:
#         clip = VideoFileClip(video_path)
#     except OSError:
#         print(f"{'ERROR '*10} : {video_path}")
#         return []

#     frames = []
#     for frame in clip.iter_frames():
#         frame = np.array(frame)
#         # Изменение размера кадра (если нужно)
#         # frame = cv2.resize(frame, target_size)
#         frames.append(frame)

#     print(f'{os.path.basename(video_path)} . . . done! Processed {len(frames)} frames.')
#     return frames
'''

### FFMPEG
'''
def process_video(video_path, target_size=(256, 256)):
    try:
        probe = ffmpeg.probe(video_path)
    except:
        print(f"{'ERROR '*10} : {video_path}")
        return []

    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])

    out, _ = (
        ffmpeg
        .input(video_path)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=True, capture_stderr=True)
    )

    video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
    frames = [frame for frame in video]

    print(f'{os.path.basename(video_path)} . . . done! Processed {len(frames)} frames.')
    return frames
'''

if __name__ == "__main__":
    parser = ArgumentParser()
    

   
    parser.add_argument("--source_video", help="path to source video")
    parser.add_argument("--driving_video", help="path to driving video")
    

    
    parser.add_argument("--open_mouth", dest="open_mouth", type=int, default=0, 
                        help="1 if opened mouth observing required, 2 if closed mouth observing required, else 0 - not observing (default 0)")

    parser.add_argument("--n", dest="n", type=int, default=1, 
                        help="number of best frames (default 1)")


 
   
 


    opt = parser.parse_args()
    
    if opt.n <= 0:
        print('--n parametr must be at least 1')

    
    
   
    print('reading source...')

    source_video = process_video(opt.source_video)
    # cv2.destroyAllWindows()
 






    print('reading driving...')
    driving_video = process_video(opt.driving_video)


    print('\nparams:')
    print('n: ', int(opt.n))
    print('mouth: ', opt.open_mouth)
    print('len source: ', len(source_video))
    print('len driving: ', len(driving_video))
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
    print('saving to txt . . .', end = '')

    if not os.path.exists("best_frames"): 
      
     
        os.makedirs("best_frames")
    with open(f'best_frames/best_frames_{os.path.basename(opt.source_video).split(".")[0]}_{os.path.basename(opt.driving_video).split(".")[0]}.txt', 'w') as file:
        for el in list_of_best_frames:
            file.write(str(el)+'\n')
    print(' OK')
    
    

