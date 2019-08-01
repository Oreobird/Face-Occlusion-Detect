# Face-Occlusion-Detect
A simple CNN face occlusion detect implemented with tensorflow keras

### Depencies
```
dlib >= 19.17.0
tensorflow >= 1.12.0
keras >= 2.2.4
numpy >= 1.11.1
scipy >= 0.14
opencv-python >= 3.4.3
```
### Usage
##### 1. Data and trained models download
download link: [https://pan.baidu.com/s/10LvoXEUGMTZjufd7R8jh4A](https://pan.baidu.com/s/10LvoXEUGMTZjufd7R8jh4A)<br>
code: 0p5j <br>
Download Cofw dataset and pretrained models in source code directory

##### 2. Train
(1) prepare data
```
    python prepare_data_cofw.py --data_dir 'cofw data directory"
```
(2) train
```
python main.py --proj_dir \
           --proj_dir "./" \   #Project directory
           --input_size 96 \   #Input image size to train
           --batch_size 100 \  #train batch size
           --fine_tune False \ #Finetune VGG16 or not
           --epochs 100 \      #Train epochs
           --train True\     #Train or test
```
##### 3. Test
(1) test on test_data
```
python main.py --proj_dir \
           --proj_dir "./" \   #Project directory
           --input_size 96 \   #Input image size to train
           --fine_tune False \ #Finetune VGG16 or not
           --train False\     #Train or test
```
(2) test on camera video stream data
Need a camera device
```
python main.py --proj_dir \
           --proj_dir "./" \   # Project directory
           --input_size 96 \   # Input image size to train
           --fine_tune False \ # Finetune VGG16 or not
           --camera_test True 
```

