# How To Drive The Project:
## Step 1: Quantize model with Vitis-AI:
1. Quantization
* Using a subset (70 images) of validation data for calibration.
```
$ python model_quant.py --quant_mode calib --subset_len 70
```

2. Export xmodel
```
$ python model_quant.py --quant_mode test --subset_len 1 --batch_size 1  --deploy
```

## Step 2: Open Docker container:
```
$ cd {VITIS_AI_PATH}
$ sudo chmod 666 /var/run/docker.sock
$ ./docker_run.sh --device /dev/video0 xilinx/vitis-ai-cpu:latest
```
![](https://i.imgur.com/UFhE4D1.png)


## Step 3: Setup enviroments:
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
```
$ export DISPLAY=":0"
$ sudo apt update
$ sudo apt-get install libcanberra-gtk-module libcanberra-gtk3-module
```
## Step 6: Demo:
```
$ cd {FOLDER_PATH}
$ bash -x build.sh
$ ./{FOLDER_NAME} dpu_HarDMSEG.xmodel {VIDEO_PATH1} {VIDEO_PATH2} {VIDEO_PATH3} {VIDEO_PATH4}
```
![](https://i.imgur.com/BR5RNX6.png)

# Result
* Use only one CPU : Intel® Core™ i7-3770
![](https://i.imgur.com/xyQadI6.png)

## Demo video

https://www.youtube.com/watch?v=YRuRyRNlGx0
