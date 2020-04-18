# Action Recognition in Video

This report will serve as my course project 4 where I investigate nueral network models for action recognition. I will attempt to detect the action 'Phoning'

I will mainly use the [DALY dataset](http://thoth.inrialpes.fr/daly/index.php ).


## Download videos


```
$ cd download_videos/     
$ apt-get install youtube-dl  # Install youtube-dl
$ bash download_videos.sh     # Downloads the DALY dataset
```

Then, moving the data file into the data file and rename is as 'videos'
```
$ cd data/
$ python3 extract_frames.py   # Extracts frames from the video
```

## ConvLSTM

 Enables action recognition in video by a bi-directional LSTM operating on frame embeddings extracted by a pre-trained ResNet-101 (ImageNet).

The model is composed of:
* A convolutional feature extractor (ResNet-101) which provides a latent representation of video frames
* A bi-directional LSTM classifier which based on the latent representation of the video predicts the activity depicted

I have made a trained model available [here](https://drive.google.com/open?id=1EWprDnL2XCGIhBW8tpx5NC-Y0eHuW5ot).

### Train  

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
$ python3 test_on_video.py  --video_path  \
                            --checkpoint_model 
```



