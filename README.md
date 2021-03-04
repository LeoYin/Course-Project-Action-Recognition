# CSCE-689-Course-Project: Action Recognition in Videos

This is the course project of Lihao Yin in CSCE 689

This report will serve as my course project 4 where I investigate neural network models for action recognition.

I used the [DALY dataset](http://thoth.inrialpes.fr/daly/index.php ).


## Download videos


```
$ cd download_videos/     
$ apt-get install youtube-dl  # Install youtube-dl
$ bash download_videos.sh     # Downloads the DALY dataset
```

Then, moving the data file into the data file and rename is as 'videos'
```
$ cd data/
# Extracts optical flow frames from the video
$ python3 extract_frames.py   --dataset_path  \
                              --annot_path daly1.1.0.pkl  # Anotation of the original videos
```

## Optical flow frames
If you want to convert the RGB videos into optical flow videos
```
$python OpticalFlow.py   --video_path
```


## ConvLSTM

 Enables action recognition in video by a bi-directional LSTM operating on frame embeddings extracted by a pre-trained ResNet-101 (ImageNet).

The model is composed of:
* A convolutional feature extractor (ResNet-101) which provides a latent representation of video frames
* A bi-directional LSTM classifier which based on the latent representation of the video predicts the activity depicted

I have made a trained model from RGB frames available [here](https://drive.google.com/open?id=1EWprDnL2XCGIhBW8tpx5NC-Y0eHuW5ot).

and  a trained model from optical flow frames available [here](https://drive.google.com/open?id=1TEj2SF22qO0Q4QRcL_2g_CO-hmAKrEc7).

### Train  
The frame data should be saved as the following
```
/data/videos-instance
    Class_Name/
    VideoClip_Name/
    Number.jpg
```
Then traning the data
```
$ python3 train.py  --dataset_path  \ 
                    --num_epochs  \
                    --batch_size \
                    --sequence_length \
                    --img_dim \
                    --latent_dim
```

### Test on Video
[Here](https://drive.google.com/open?id=1TmOUmDIZbuXJPQp9nxIS-KR_pBLZxxrO) is the link for five test video samples.

Download the testing videos into the file test/test_videos/, and download the tained model into the file model_checkpoints/

```
$ python3 test_on_video.py  --video_path  \  # the files including the test videos (mp4 format)
                            --save_path  \  # 
                            --checkpoint_model1  \  # the trained model from RGB frames
                            --checkpoint_model2  \  # the trained model from optical flow frames
```




