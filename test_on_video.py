from models import *
from dataset import *
import argparse
import os
import glob
import tqdm
from torchvision.utils import make_grid
from PIL import Image, ImageDraw
import skvideo.io
import ssl
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import av


def extract_frames(video_path):
    frames = []
    video = av.open(video_path)
    for frame in video.decode(0):
        yield frame.to_image()

if __name__ == "__main__":
    ssl._create_default_https_context = ssl._create_unverified_context
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="test/test_videos", help="Path to video")
    parser.add_argument("--save_path", type=str, default="test/test_results", help="Path to save results")
    parser.add_argument("--image_dim", type=int, default=112, help="Height / width dimension")
    parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
    parser.add_argument("--latent_dim", type=int, default=512, help="Dimensionality of the latent representation")
    parser.add_argument("--checkpoint_model1", type=str, default="model_checkpoints/ConvLSTM_20.pth", help="Optional path to checkpoint model")
    parser.add_argument("--checkpoint_model2", type=str, default="model_checkpoints/ConvLSTM_Flow.pth", help="Optional path to checkpoint model")

    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (opt.channels, opt.image_dim, opt.image_dim)

    transform = transforms.Compose(
        [
            transforms.Resize(input_shape[-2:], Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    labels = ['Phoning','ApplyingMakeUpOnLips','BrushingTeeth','CleaningFloor','CleaningWindows','Drinking','FoldingTextile','Ironing','PlayingHarmonica','TakingPhotosOrVideos']
    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)

    model1 = ConvLSTM(num_classes=len(labels), latent_dim=opt.latent_dim)
    model1.to(device)
    model1.load_state_dict(torch.load(opt.checkpoint_model2, map_location=torch.device('cpu')))
    model1.eval()

            
    # Define model and load model checkpoint
    model2 = ConvLSTM(num_classes=len(labels), latent_dim=opt.latent_dim)
    model2.to(device)
    model2.load_state_dict(torch.load(opt.checkpoint_model1, map_location=torch.device('cpu')))
    model2.eval()

    # Extract predictions
    for video in glob.glob(os.path.join(opt.video_path,'*.mp4')):
        output_frames = []
        y = []
        record = []
        cap=cv2.VideoCapture(video)
        fps = cap.get(5)
        video_name = video.split('.mp4')[0].split('/')[-1]
    
        for j, frame in enumerate(tqdm.tqdm(extract_frames(video), desc="Processing frames")):
            if j == 0:
                first_frame = np.array(frame)
                resize_dim = 600
                max_dim = max(first_frame.shape)
                scale = resize_dim/max_dim
                first_frame = cv2.resize(first_frame, None, fx=scale, fy=scale)
                prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
                mask = np.zeros_like(first_frame)
                mask[..., 1] = 255
            else:
                cur_frame = np.array(frame)
                gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, None, fx=scale, fy=scale)
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale = 0.5, levels = 5, winsize = 11, iterations = 5, poly_n = 5, poly_sigma = 1.1, flags = 0)
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                mask[..., 0] = angle * 180 / np.pi / 2
                mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
                rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
                cur_frame = cv2.resize(cur_frame, None, fx=scale, fy=scale)
                dense_flow = cv2.addWeighted(cur_frame, 1,rgb, 2, 0)
                dense_flow = Image.fromarray(dense_flow.astype('uint8')).convert('RGB')
                image_tensor1 = Variable(transform(dense_flow)).to(device)
                image_tensor1 = image_tensor1.view(1, 1, *image_tensor1.shape)
                prev_gray = gray

            image_tensor2 = Variable(transform(frame)).to(device)
            image_tensor2 = image_tensor2.view(1, 1, *image_tensor2.shape)

            # Get label prediction for frame
            with torch.no_grad():
                prediction = model2(image_tensor2)
                if j != 0:
                    prediction1 = model1(image_tensor1)
                    prediction = (prediction+prediction1)/2
                label_id = prediction.argmax(1).item()
                if label_id == 0:
                    predicted_label = 'Phoning'
                    record.append([j/fps, torch.max(prediction).item()])
                    y.append('Phoning')
                else:
                    predicted_label = ''
                    y.append('Other')

            # Draw label on frame
            d = ImageDraw.Draw(frame)
            d.text(xy=(10, 10), text=predicted_label, fill=(255, 255, 255))
            
            output_frames += [frame]

        # Create video from frames
        writer = skvideo.io.FFmpegWriter(os.path.join(opt.save_path,video_name+'.avi'))
        for frame in tqdm.tqdm(output_frames, desc="Writing to video"):
            writer.writeFrame(np.array(frame))
        writer.close()

        record = {'Phoning': record}
        json_str = json.dumps(record)
        with open(opt.save_path+'/timeLabel_'+video_name+'.json', 'w') as json_file:
            json_file.write(json_str)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_title('Plot '+video_name)
        plt.xlabel('Time')
        plt.ylabel('Label')
        x = np.arange(0,len(y))/fps
        ax1.scatter(x,y,c = 'r',marker = 'o')
        plt.savefig(opt.save_path+'/'+video_name+'_plot.png',bbox_inches='tight')
