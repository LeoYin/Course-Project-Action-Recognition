"""
Helper script for extracting frames from the UCF-101 dataset
"""

#import av
import glob
import os
import time
import tqdm
import datetime
import argparse
import pickle
import cv2
import numpy as np
from PIL import Image

def extract_frames(video_path):
    frames = []
    vc = cv2.VideoCapture(video_path)
    ret, frame = vc.read()
    while(ret):
        yield frame
        ret, frame = vc.read()
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    #video = av.open(video_path)
    #for frame in video.decode(0):
    #    yield frame.to_image()
    

prev_time = time.time()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="videos", help="Path to video dataset")
    parser.add_argument("--annot_path", type=str, default="daly1.1.0.pkl", help="Path to frames")
    parser.add_argument("--skip_frames", type=int, default=20, help="Remaining part frames")

    opt = parser.parse_args()
    print(opt)

    with open(opt.annot_path, "rb") as f:
        daly = pickle.load(f, encoding='latin1')
        
    time_left = 0
    video_paths = glob.glob(os.path.join(opt.dataset_path, "*.mp4"))
    print(len(video_paths))
    for i, video_path in enumerate(video_paths):
        sequence_id = video_path[-15:]
        sequence_types = daly['annot'][sequence_id]['annot'].keys()
        sequence_name = sequence_id.split(".mp4")[0]
        sequence_dct = {}
        lag_time = []
        for seq_type in sequence_types:
            n_instance = len(daly['annot'][sequence_id]['annot'][seq_type])
            for ii in range(n_instance):
                beginTime = daly['annot'][sequence_id]['annot'][seq_type][ii]['beginTime']
                endTime = daly['annot'][sequence_id]['annot'][seq_type][ii]['endTime']
                begin = beginTime*daly['annot'][sequence_id]['annot'][seq_type][ii]['keyframes'][0]['frameNumber']/daly['annot'][sequence_id]['annot'][seq_type][ii]['keyframes'][0]['time']
                end = endTime*daly['annot'][sequence_id]['annot'][seq_type][ii]['keyframes'][0]['frameNumber']/daly['annot'][sequence_id]['annot'][seq_type][ii]['keyframes'][0]['time']
                begin = int(begin)
                end = int(end)
                skip = (end - begin) // 200 + 1
                sequence_path = os.path.join(f"{opt.dataset_path}-instance", seq_type, sequence_name+str(ii))
                if not os.path.exists(sequence_path):
                    sequence_dct[begin] = {'end':end, 'type':seq_type, 'seq_path':sequence_path, 'i':0, 'skip':skip}
                    lag_time.append(begin)      

        
        #video = av.open(video_path)
        #f = list(video.decode(0))[0]
        #if type(f)==av.audio.frame.AudioFrame:
        #    continue

        lag_time.sort()
        for t in lag_time:
            os.makedirs(sequence_dct[t]['seq_path'], exist_ok=True)
        
        print(video_path+'\r')
        # Extract frames

        is_work = False
        t = 0
        n_end = len(lag_time)
        for j, frame in enumerate(
            tqdm.tqdm(
                extract_frames(video_path),
                desc=f"[{i}/{len(video_paths)}] {sequence_name} : ETA {time_left}",
            )
        ):
            if is_work:
                if (j%sequence_dct[lag_time[t]]['skip'])==0:
                    id = sequence_dct[lag_time[t]]['i']
                    #frame = np.array(frame)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, None, fx=scale, fy=scale)
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale = 0.5, levels = 5, winsize = 11, iterations = 5, poly_n = 5, poly_sigma = 1.1, flags = 0)
                    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    mask[..., 0] = angle * 180 / np.pi / 2
                    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
                    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
                    frame = cv2.resize(frame, None, fx=scale, fy=scale)
                    dense_flow = cv2.addWeighted(frame, 1,rgb, 2, 0)
                    dense_flow = Image.fromarray(dense_flow.astype('uint8')).convert('RGB')
                    dense_flow.save(os.path.join(sequence_dct[lag_time[t]]['seq_path'], f"{id}.jpg"))
                    prev_gray = gray
                    sequence_dct[lag_time[t]]['i'] += 1
                if j == sequence_dct[lag_time[t]]['end']:
                    is_work = False
                    t += 1
                    if t == n_end:
                        break
            else:
                if j == lag_time[t]:
                    is_work = True
                    first_frame = frame
                    resize_dim = 600
                    #first_frame = np.array(first_frame)
                    max_dim = max(first_frame.shape)
                    scale = resize_dim/max_dim
                    first_frame = cv2.resize(first_frame, None, fx=scale, fy=scale)
                    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
                    mask = np.zeros_like(first_frame)
                    mask[..., 1] = 255
                    #frame.save(os.path.join(sequence_dct[lag_time[t]]['seq_path'], f"{sequence_dct[lag_time[t]]['i']}.jpg"))
                    sequence_dct[lag_time[t]]['i'] += 1


        # Determine approximate time left
        videos_left = len(video_paths) - (i + 1)
        time_left = datetime.timedelta(seconds=videos_left * (time.time() - prev_time))
        prev_time = time.time()
