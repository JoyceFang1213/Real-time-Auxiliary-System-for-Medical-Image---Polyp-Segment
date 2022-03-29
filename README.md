# How To Run The Project:

## Step 1: Open Docker container
NOTE: If you have a compatible Nvidia graphics card with CUDA support, you may install the GPU docker. Remember to run the GPU docker with replacement of vitis-ai-gpu:latest.
```
$ cd {VITIS_AI_PATH}
$ sudo chmod 666 /var/run/docker.sock
$ ./docker_run.sh --device /dev/video0 xilinx/vitis-ai-cpu:latest
```
![](https://i.imgur.com/UFhE4D1.png)

## Step 2: Quantize model with Vitis-AI:
1. Quantization
* Using a subset (70 images) of validation data for calibration.
```
$ python model_quant.py --quant_mode calib --subset_len 70
```

2. Export xmodel
```
$ python model_quant.py --quant_mode test --subset_len 1 --batch_size 1  --deploy
```


## Step 3: Setup environments:
1. Setup VCK5000
```
$ cd setup/vck5000
$ source ./setup.sh
```
2. Conda Pytorch enviroments
```
$ conda activate vitis-ai-pytorch
$ source ./setup.sh
```
3. Check DPU
```
$ sudo chmod o=rw /dev/dri/render*
$ xdputil query
```
![](https://i.imgur.com/tiaquLA.png)


## Step 4: Vitis-AI compilation:
```
$ cd /workspace/
$ vai_c_xir -x HarDMSEG_int.xmodel -a arch.json -o ./ -n dpu_HarDMSEG
```

## Step 5: Install necesarry package:
These packages are for showing the windows on local screens.
```
$ export DISPLAY=":0"
$ sudo apt update
$ sudo apt-get install libcanberra-gtk-module libcanberra-gtk3-module
```
## Step 6: Demo:
Note: At most 8 videos are supported due to the limitation of DPU.
```
$ cd {FOLDER_PATH}
$ bash -x build.sh
$ ./{FOLDER_NAME} dpu_HarDMSEG.xmodel {VIDEO_PATH1} {VIDEO_PATH2} {VIDEO_PATH3} {VIDEO_PATH4}
```
![](https://i.imgur.com/aca0EHN.jpg)


# Result
* Use only one CPU : Intel® Core™ i7-3770

![](https://i.imgur.com/xyQadI6.png)

