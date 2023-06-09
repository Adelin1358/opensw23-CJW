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
* 

See [transforms folder](https://github.com/rrmina/fast-neural-style-pytorch/tree/master/transforms) for some pretrained weights. For more pretrained weights, see my [Gdrive](https://drive.google.com/open?id=1m9g1PP7gPo-jPfRDxzdGozMzftu3az6P) or [Dropbox](https://www.dropbox.com/sh/066lk1m5sgkhtmi/AAAqVwNhCHsrK2p8Xil1ftH4a?dl=0).



### Stylizing Images

**`stylize.py`**: Loads a pre-trained transformer network weight and applies style (1) to a content image or (2) to the images inside a folder

```

python stylize.py

```

**Options**

* `STYLE_TRANSFORM_PATH`: path of the pre-trained weights of the the transformation network. Sample pre-trained weights are availabe in `transforms` folder, including their implementation parameters.

* `PRESERVER_COLOR`: set to `True` if you want to preserve the original image's color after applying style transfer. Default value is `False`





### Stylizing Webcam

**`webcam.py`**: Captures and saves webcam output image, perform style transfer, and again saves a styled image. Reads the styled image and show in window. 

```

python webcam.py

```

**Options**

* `STYLE_TRANSFORM_PATH`: pretrained weight of the style of the transformation network to use for video style transfer. Default is `"transforms/aggressive.pth"`

* `WIDTH`: width of the webcam output window. Default is `1280`

* `HEIGHT`: height of the webcam output window. Default is `720`
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



-변형시킬 이미지를 images/ 파일에서 선정한 뒤, 적용할 스타일을 transforms/ 파일에 PTH파일를 골라 적용한다.
  <p align = 'center'>
  <img src = 'course/4.png' height = '250px'>
  <img src = 'course/5.png' height = '250px'>
  </p>

  
  
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


<p align = 'center'>
<img src = 'results2/1.png' height = '400px'>
<img src = 'results2/２.png' height = '400px'> 
<img src = 'results2/３.png' height = '400px'>  

  


#### 1-3. 화풍이 유사한 여러 이미지를 활용해 훈련시킨 케이스
 
  
<p align = 'center'>
<img src = 'results2/9.png' height = '400px'>
<img src = 'results2/8.png' height = '400px'> 

 
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
<p align = 'center'>
<img src = 'results2/6.png' height = '400px'>
  

- splatoon.jpg 이미지를 약 2000장 훈련 시켰을 때, 로스

<img src = 'results2/501 - 복사본.png' height = '300px'>



- 같은 화풍의 penguin 이미지들을 약 2000장 훈련 시켰을 때, 로스

<img src = 'results2/502.png' height = '300px'>


- 결론 : [1-2] 케이스와 [1-3] 케이스의 Total Loss를 비교해보면 수치적으로는 [1-3] '화풍이 유사한 여러 이미지를 활용해 훈련시킨 케이스'가 Loss가 적은 것으로 나타났다. 그러나 상대적으로 이 차이가 미미한 지, 큰 차이가 있는지 알 수 없어 여러 이미지를 활용하는 것이 더 효율이 좋다고 말하는 것은 섣부른 판단이라고 생각한다. 더 많은 케이스를 비교하고, 적절한 변인통제가 필요할 것으로 보인다.


### 주의사항
-train.py 주의사항


훈련이 끝난 후 나오는 로스율 그래프 오류 존재 => 무시할 것
<p align = 'center'>
<img src = 'results2/penguin(502).png' height = '400px'>



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


## Presentation
https://www.youtube.com/watch?v=RM0_GDxJLnM




  
