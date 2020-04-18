from models import *
from dataset import *
from data.extract_frames import extract_frames
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

if __name__ == "__main__":
    ssl._create_default_https_context = ssl._create_unverified_context
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="test/test_videos", help="Path to video")
    parser.add_argument("--save_path", type=str, default="test/test_results", help="Path to save results")
    parser.add_argument("--image_dim", type=int, default=112, help="Height / width dimension")
    parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
    parser.add_argument("--latent_dim", type=int, default=512, help="Dimensionality of the latent representation")
    parser.add_argument("--checkpoint_model", type=str, default="model_checkpoints/ConvLSTM_20.pth", help="Optional path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    assert opt.checkpoint_model, "Specify path to checkpoint model using arg. '--checkpoint_model'"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (opt.channels, opt.image_dim, opt.image_dim)

    transform = transforms.Compose(
        [
            transforms.Resize(input_shape[-2:], Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    #labels = sorted(list(set(os.listdir(opt.dataset_path))))
    labels = ['Phoning','ApplyingMakeUpOnLips','BrushingTeeth','CleaningFloor','CleaningWindows','Drinking','FoldingTextile','Ironing','PlayingHarmonica','TakingPhotosOrVideos']

    # Define model and load model checkpoint
    model = ConvLSTM(num_classes=len(labels), latent_dim=opt.latent_dim)
    model.to(device)
    model.load_state_dict(torch.load(opt.checkpoint_model, map_location=torch.device('cpu')))
    model.eval()

    # Extract predictions
    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)
    for video in glob.glob(os.path.join(opt.video_path,'*.mp4')):
        record = []
        output_frames = []
        video_name = video.split('.mp4')[0].split('/')[-1]

        cap=cv2.VideoCapture(video)
        fps = cap.get(5)

        y = []
    
        print(video_name)
        for j, frame in enumerate(tqdm.tqdm(extract_frames(video), desc="Processing frames")):
            image_tensor = Variable(transform(frame)).to(device)
            image_tensor = image_tensor.view(1, 1, *image_tensor.shape)

            # Get label prediction for frame
            with torch.no_grad():
                prediction = model(image_tensor)
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
            d.text(xy=(20, 20), text=predicted_label, fill=(255, 255, 255))

            output_frames += [frame]

        # Create video from frames
        #height, width = output_frames[0].size
        #size = (width,height)
        #out = cv2.VideoWriter(opt.save_path+video_name+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        #for i in range(len(output_frames)):
        #    out.write(output_frames[i])
        #out.release()
        writer = skvideo.io.FFmpegWriter(os.path.join(opt.save_path,video_name+'.avi'))
        for frame in tqdm.tqdm(output_frames, desc="Writing to video"):
            writer.writeFrame(np.array(frame))
        writer.close()

        record = {'Phoning': record}
        json_str = json.dumps(record)
        with open(opt.save_path+'/timeLabel.json', 'w') as json_file:
            json_file.write(json_str)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_title('Plot '+video_name)
        plt.xlabel('Time')
        plt.ylabel('Label')
        x = np.arange(0,len(y))/fps
        ax1.scatter(x,y,c = 'r',marker = 'o')
        plt.savefig(opt.save_path+'/'+video_name+'_plot.png',bbox_inches='tight')
    
    
