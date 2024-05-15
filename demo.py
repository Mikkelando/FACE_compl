import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import modules.generator as GEN
import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
from scipy.spatial import ConvexHull
from collections import OrderedDict
import pdb
if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.load(f)
    if opt.kp_num != -1:
        config['model_params']['common_params']['num_kp'] = opt.kp_num
    generator = getattr(GEN, opt.generator)(**config['model_params']['generator_params'],**config['model_params']['common_params'],**{'mbunit':opt.mbunit,'mb_spatial':opt.mb_spatial,'mb_channel':opt.mb_channel})
    if not cpu:
        generator.cuda()
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()
    
    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path,map_location="cuda:0")
    
    ckp_generator = OrderedDict((k.replace('module.',''),v) for k,v in checkpoint['generator'].items())
    generator.load_state_dict(ckp_generator)
    ckp_kp_detector = OrderedDict((k.replace('module.',''),v) for k,v in checkpoint['kp_detector'].items())
    kp_detector.load_state_dict(ckp_kp_detector)
    
    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()
    
    return generator, kp_detector


def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
    sources = []
    drivings = []
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)

        kp_source = kp_detector(source)
        if not cpu:
            kp_driving_initial = kp_detector(driving[:, :, 0].cuda())
        else:
            kp_driving_initial = kp_detector(driving[:, :, 0])
        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
            drivings.append(np.transpose(driving_frame.data.cpu().numpy(), [0, 2, 3, 1])[0])
            sources.append(np.transpose(source.data.cpu().numpy(), [0, 2, 3, 1])[0])
            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return sources, drivings, predictions

def find_best_frame(source, driving, cpu=False):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num



def find_n_best_frames(source, driving, n=1, cpu=False, mouth=False):
    import face_alignment
    
    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norms = []
    frame_nums = []
    for i, image in tqdm(enumerate(driving)):

        if check_eye(image) > 0.5:
            if mouth == True:
                if check_mouth(image) > 0.79: #THRESHOLD
                    kp_driving = fa.get_landmarks(255 * image)[0]
                    kp_driving = normalize_kp(kp_driving)
                    new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
                    norms.append(new_norm)
                    frame_nums.append(i)
            else:
                kp_driving = fa.get_landmarks(255 * image)[0]
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



# def check_eye(driving_frame):
#     '''
#     return confidence of opened eyes on frame
#     '''
#     ...
#     return 0.5

def check_mouth(driving_frame):
    '''
    return confidence of opened mouth on frame
    '''
    ...
    
    
    return 0.5







if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='vox-cpk.pth.tar', help="path to checkpoint to restore")

    parser.add_argument("--source_image", default='sup-mat/source.png', help="path to source image")
    parser.add_argument("--driving_video", default='sup-mat/source.png', help="path to driving video")
    parser.add_argument("--result_video", default='result.mp4', help="path to output")
    
    parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")
    parser.add_argument("--generator", type=str, required=True)
    parser.add_argument("--kp_num", type=int, required=True)
    parser.add_argument("--mb_channel",type=int, default=512, help='depth mode')
    parser.add_argument("--mb_spatial",type=int, default=32, help='depth mode')
    parser.add_argument("--mbunit",type=str, default='', help='depth mode')
    parser.add_argument("--memsize",type=int, default=1, help='depth mode')

    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true", 
                        help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")
    

    parser.add_argument("--find_n_best_frame", dest="find_n_best_frame", action="store_true", 
                        help="n bests frames to start (only for print)")
    
    parser.add_argument("--open_mouth", dest="open_mouth", type=int, default=0, 
                        help="1 if open mouth observing required, else 0 (default 0)")

    parser.add_argument("--n", dest="n", type=int, default=1, 
                        help="number of best frames (default 1)")

    parser.add_argument("--best_frame", dest="best_frame", type=int, default=None,  
                        help="Set frame to start from.")
 
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
 

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

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
    generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)

    if opt.find_best_frame or opt.best_frame is not None or opt.find_n_best_frame :
        if opt.best_frame is not None:
            i = opt.best_frame
        elif opt.find_best_frame:
            i = find_best_frame(source_image, driving_video, cpu=opt.cpu)
        elif opt.find_n_best_frame:
        
            list_of_best_frames = find_n_best_frames(source_image, driving_video, cpu=opt.cpu, n=int(opt.n), mouth = opt.open_mouth)
            if list_of_best_frames is None:
                print('BEST FRAME NOT FOUND')
                i = 0
            else:
                i = list_of_best_frames[-1]
                print('LIST OF BEST FRAMES:  ') 
                print(*list_of_best_frames)
        print ("Best frame: " + str(i))
        driving_forward = driving_video[i:]
        driving_backward = driving_video[:(i+1)][::-1]
        sources_forward, drivings_forward, predictions_forward = make_animation(source_image, driving_forward, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
        sources_backward, drivings_backward, predictions_backward = make_animation(source_image, driving_backward, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
        predictions = predictions_backward[::-1] + predictions_forward[1:]
        sources = sources_backward[::-1] + sources_forward[1:]
        drivings = drivings_backward[::-1] + drivings_forward[1:]
    else:
        sources, drivings, predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
    imageio.mimsave(opt.result_video, [np.concatenate((img_as_ubyte(s),img_as_ubyte(d),img_as_ubyte(p)),1) for (s,d,p) in zip(sources, drivings, predictions)], fps=fps)

