## Team Introduction
최정우(CHOI JUNG WOO)_202211388_Project Director


## Topic Introduction
- 개요

  이 프로그램은 이미지 스타일 변환(fast-neural-style-transfer)을 수행하는 것을 목표로 합니다. 
  
  스타일 변환은 하나의 이미지의 스타일을 다른 이미지에 적용하여 새로운 이미지를 생성하는 작업입니다. 이 프로그램은 사전 훈련된 변환 네트워크를 사용하여 이미지의 스타일을 변환하고, 결과 이미지를 저장합니다.
  
  
  이 프로그램은 다음과 같은 기능을 제공합니다:
  
  - train.py: 스타일 변환 네트워크를 학습시키는 기능으로, 주어진 스타일 이미지를 기반으로 변환 네트워크의 가중치를 조정합니다.
  
  - stylize.py: 단일 이미지나 이미지 폴더에 스타일을 적용하는 기능으로, 사전 훈련된 변환 네트워크를 사용하여 이미지를 스타일링합니다.
  
  - video.py: 비디오 파일에서 프레임을 추출하고, 추출한 프레임에 스타일을 적용하여 새로운 비디오를 생성하는 기능입니다.
  
  - webcam.py: 웹캠에서 실시간으로 비디오를 캡처하고, 캡처된 프레임에 스타일을 적용하여 화면에 출력하는 기능을 제공합니다.
 
  이 프로그램을 사용하면 스타일 전이를 쉽게 수행하고 이미지와 비디오에 다양한 스타일을 적용할 수 있습니다.
  
  
## Results
### 1.stylize.py
#### 1-1. 제공된 pth 파일을 적용한 케이스
  
  
  - up-diliman.jpg 원본
<img src = 'images/up-diliman.jpg' height = '500px'>
  
  
  - STYLE_TRANSFORM_PATH = "transforms/udnie.pth"일 경우
<img src = 'images/results/oble_udnie.jpg' height = '250px'>
  
  
  - STYLE_TRANSFORM_PATH = "transforms/mosaic.pth"일 경우
<img src = 'images/results/oble_mosaic.jpg' height = '250px'>
  
  
  - STYLE_TRANSFORM_PATH = "transforms/tokyo_ghoul.pth"일 경우
<img src = 'images/results/oble_ghoul.jpg' height = '250px'>
 
  
  - STYLE_TRANSFORM_PATH = "transforms/wave.pth" 일 경우
<img src = 'images/results/oble_wave.jpg' height = '250px'>
</p>
  
  
#### 1-2. 한 종류의 이미지를 활용해 훈련시킨 케이스

- 원본 이미지 파일
<p align = 'center'>
<img src = 'results2/splatoon.jpg' height = '400px'>
  
  
해당 이미지 선정 :
</p>


- 해당 이미지를 1장만 훈련 시켰을 경우(train.py)
<p align = 'center'>
<img src = 'results2/splatoon_1.png' height = '250px'>
</p>


- 해당 이미지를 약 150장 훈련 시켰을 경우(train.py)
<p align = 'center'>
<img src = 'results2/splatoon(38).png' height = '250px'>
</p>


- 해당 이미지를 약 800장 훈련 시켰을 경우(train.py)
<p align = 'center'>
<img src = 'results2/splatoon_194.png' height = '250px'>
</p>


- 해당 이미지를 약 2000장 훈련 시켰을 경우(train.py)
<p align = 'center'>
<img src = 'results2/splatoon(500).png' height = '400px'>
</p>


#### 1-2. 화풍이 유사한 여러 이미지를 활용해 훈련시킨 케이스
- 원본 이미지 파일
 (출처: )
<p align = 'center'>
<img src = 'results2/penguin1.jpg' height = '200px'>
<img src = 'results2/penguin2.jpg' height = '200px'>
<img src = 'results2/penguin3.jpg' height = '200px'>
<img src = 'results2/penguin4.jpg' height = '200px'>
<img src = 'results2/penguin5.jpg' height = '200px'>
  
  
해당이미지 선정 사유:


- 각각의 이미지를 400장씩, 총 2000 훈련 시켰을 경우(train.py)
<p align = 'center'>
<img src = 'results2/penguin502.png' height = '400px'>
</p>


### video.py 
#### Video Stylization
<p align = 'center'>
<a href="https://www.youtube.com/watch?v=dB7DRsnkE3g&list=PL3freW_f-7aWsJrHTG5AKpY9TPWZgnNcm">
<img src="images/results/video.gif" height = '360px'>
</a>
</p>
<p align = 'center'>
It took 6 minutes and 43 seconds to stylize a 2:11 minute-24 fps-1280x720 video on a GTX 1080 Ti. 
</p>

More videos in this [Youtube playlist](https://www.youtube.com/watch?v=dB7DRsnkE3g&list=PL3freW_f-7aWsJrHTG5AKpY9TPWZgnNcm). Unfortunately, Youtube's compression isn't friendly with style transfer videos, possibily because each frame is shaky with respect to its adjacent frames, hence obvious loss in video quality. `Raw and lossless output video can be downloaded in my` [Dropbox folder](https://www.dropbox.com/sh/ynlie98f1lb4csz/AAC4utgi8HrS_D7XDU-9FXoAa?dl=0), or [Gdrive Folder](https://drive.google.com/open?id=1uplUkayaTiThURmQTAuUqeAKuQeLxXCc)
 
  
### webcam.py 

  - STYLE_TRANSFORM_PATH = "transforms/mosaic.pth"일 경우
<img src = 'images/results/mosaic.png' height = '250px'>
  
  
  - STYLE_TRANSFORM_PATH = "transforms/tokyo_ghoul.pth"일 경우
<img src = 'images/results/tokyo.png' height = '250px'> 
  
  
  - STYLE_TRANSFORM_PATH = "transforms/wave.pth"일 경우
<img src = 'images/results/wave.png' height = '250px'> 
  
  
  - STYLE_TRANSFORM_PATH = "transforms/starry.pth"일 경우
<img src = 'images/results/starry.png' height = '250px'>   
</p>


## Analysis/Visualization



- splatoon.jpg 이미지를 약 800장 훈련 시켰을 때, 로스율 그래프
<p align = 'center'>
<img src = 'results2/splatoon(194).png' height = '250px'>
</p>


- splatoon.jpg 이미지를 약 2000장 훈련 시켰을 때, 로스율 그래프
<p align = 'center'>
<img src = 'splatoon(501)_loss.png' height = '400px'>
</p>


- penguin 이미지들을 약 2000장 훈련 시켰을 때, 로스율 그래프
<p align = 'center'>
<img src = 'results2/penguin(502).png' height = '400px'>
</p>


## Installation


### Requirements
Most of the codes here assume that the user have access to CUDA capable GPU, at least a GTX 1050 ti or a GTX 1060


#### Data Files
* [Pre-trained VGG16 network weights](https://github.com/jcjohnson/pytorch-vgg) - put it in `models/` directory
* [MS-COCO Train Images (2014)](http://cocodataset.org/#download) - 13GB - put `train2014` directory in `dataset/` directory
* [torchvision](https://pytorch.org/) - `torchvision.models` contains the VGG16 and VGG19 model skeleton


#### Dependecies
* [PyTorch](https://pytorch.org/)
* [opencv2](https://matplotlib.org/users/installing.html)
* [NumPy](https://www.scipy.org/install.html)
* [FFmpeg](https://www.ffmpeg.org/) (Optional) - Installation [Instruction here](https://github.com/adaptlearning/adapt_authoring/wiki/Installing-FFmpeg)


#### Usage
All arguments, parameters and options are **`hardcoded`** inside these 5 python files. **Before using the codes, please arrange your files and folders [as defined below](#files-and-folder-structure)**.
## Training Style Transformation Network
**`train.py`**: trains the transformation network that learns the style of the `style image`. Each model in `transforms` folder was trained for roughly 23 minutes, with single pass (1 epoch) of 40,000 training images, and a batch size of 4, on a GTX 1080 Ti. 
```
python train.py
```

**Options**
* `TRAIN_IMAGE_SIZE`: sets the dimension (height and weight) of training images. Bigger GPU memory is needed to train with larger images. Default is `256`px.
* `DATASET_PATH`: folder containing the MS-COCO `train2014` images. Default is `"dataset"` 
* `NUM_EPOCHS`: Number of epochs of training pass. Default is `1` with 40,000 training images
* `STYLE_IMAGE_PATH`: path of the style image
* `BATCH_SIZE`: training batch size. Default is 4 
* `CONTENT_WEIGHT`: Multiplier weight of the loss between content representations and the generated image. Default is `8`
* `STYLE_WEIGHT`: Multiplier weight of the loss between style representations and the generated image. Default is `50`
* `ADAM_LR`: learning rate of the adam optimizer. Default is `0.001`
* `SAVE_MODEL_PATH`: path of pretrained-model weights and transformation network checkpoint files. Default is `"models/"`
* `SAVE_IMAGE_PATH`: save path of sample tranformed training images. Default is `"images/out/"`
* `SAVE_MODEL_EVERY`: Frequency of saving of checkpoint and sample transformed images. 1 iteration is defined as 1 batch pass. Default is `500` with batch size of `4`, that is 2,000 images
* `SEED`: Random seed to keep the training variations as little as possible

**`transformer.py`**: contains the architecture definition of the trasnformation network. It includes 2 models, `TransformerNetwork()` and `TransformerNetworkTanh()`. `TransformerNetwork` doesn't have an extra output layer, while `TransformerNetworkTanh`, as the name implies, has for its output, a Tanh layer and a default `output multiplier of 150`. `TransformerNetwork` faithfully copies the style and colorization of the style image, while Tanh model produces images with darker color; which brings a **`retro style effect`**.

**Options** 
* `norm`: sets the normalization layer to either Instance Normalization `"instance"` or Batch Normalization `"batch"`. Default is `"instance"`
* `tanh_multiplier`: output multiplier of the Tanh model. The bigger the number, the bright the image. Default is `150`

**`experimental.py`**: contains the model definitions of the experimental transformer network architectures. These experimental transformer networks largely borrowed ideas from the papers [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431) or more commonly known as `ResNeXt`, and [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) or more commonly known as `DenseNet`. These experimental networks are designed to be lightweight, with the goal of minimizing the compute and memory needed for better real-time performance. 

See [table below for the comparison of different transformer networks](#comparison-of-different-transformer-networks).

See [transforms folder](https://github.com/rrmina/fast-neural-style-pytorch/tree/master/transforms) for some pretrained weights. For more pretrained weights, see my [Gdrive](https://drive.google.com/open?id=1m9g1PP7gPo-jPfRDxzdGozMzftu3az6P) or [Dropbox](https://www.dropbox.com/sh/066lk1m5sgkhtmi/AAAqVwNhCHsrK2p8Xil1ftH4a?dl=0).

### Stylizing Images
**`stylize.py`**: Loads a pre-trained transformer network weight and applies style (1) to a content image or (2) to the images inside a folder
```
python stylize.py
```
**Options**
* `STYLE_TRANSFORM_PATH`: path of the pre-trained weights of the the transformation network. Sample pre-trained weights are availabe in `transforms` folder, including their implementation parameters.
* `PRESERVER_COLOR`: set to `True` if you want to preserve the original image's color after applying style transfer. Default value is `False`

### Stylizing Videos
**`video.py`**: Extracts all frames of a video, apply fast style transfer on each frames, and combine the styled frames into an output video. The output video doesn't retain the original audio. Optionally, you may use FFmpeg to merge the output video and the original video's audio.
```
python video.py
```
**Options**
* `VIDEO_NAME`: path of the original video
* `FRAME_SAVE_PATH`: parent folder of the save path of the extracted original video frames. Default is `"frames/"`
* `FRAME_CONTENT_FOLDER`: folder of the save path of the extracted original video frames. Default is `"content_folder/"`
* `FRAME_BASE_FILE_NAME`: base file name of the extracted original video frames.  Default is  `"frame"`
* `FRAME_BASE_FILE_TYPE`: save image file time ".jpg"
* `STYLE_FRAME_SAVE_PATH`: path of the styled frames. Default is `"style_frames/"`
* `STYLE_VIDEO_NAME`: name(or save path) of the output styled video. Default is `"helloworld.mp4"`
* `STYLE_PATH`: pretrained weight of the style of the transformation network to use for video style transfer. Default is `"transforms/aggressive.pth"`
* `BATCH_SIZE`: batch size of stylization of extracted original video frames. A 1080ti 11GB can handle a batch size of 20 for 720p videos, and 80 for a 480p videos. Dafult is `1`
* `USE_FFMPEG`(Optional): Set to `True` if you want to use FFmpeg in extracting the original video's audio and encoding the styled video with the original audio.

### Stylizing Webcam
**`webcam.py`**: Captures and saves webcam output image, perform style transfer, and again saves a styled image. Reads the styled image and show in window. 
```
python webcam.py
```
**Options**
* `STYLE_TRANSFORM_PATH`: pretrained weight of the style of the transformation network to use for video style transfer. Default is `"transforms/aggressive.pth"`
* `WIDTH`: width of the webcam output window. Default is `1280`
* `HEIGHT`: height of the webcam output window. Default is `720`


</p>

### course

-Data Files 설치 후 파일 상황
  <p align = 'center'>
  <img src = 'course/1.png' height = '250px'>
  </p>



-stylize.py 주의사항
  21 line 주석처리된 부분=>TYLE_TRANSFORM_PATH =tokyo_ghoul.pth 일 경우 
  18줄 코드 주석 처리후, 밑 코드를 실행시킬 것!!!
  
  <p align = 'center'>
  <img src = 'course/2.png' height = '250px'>
  
  </p>


-webcam.py 주의사항
  25 line 주석처리된 부분=>TYLE_TRANSFORM_PATH =tokyo_ghoul.pth 일 경우 
  24 line 코드 주석 처리후, 26 line 코드를 실행시킬 것!!!
  <p align = 'center'>
  <img src = 'course/3.png' height = '250px'>
  </p>



-변형시킬 이미지를 images/ 파일에서 선정한 뒤, 적용할 스타일을 transforms/ 파일에 PTH파일를 골라 적용한다.
  <p align = 'center'>
  <img src = 'course/4.png' height = '250px'>
  <img src = 'course/5.png' height = '250px'>
  </p>








  
