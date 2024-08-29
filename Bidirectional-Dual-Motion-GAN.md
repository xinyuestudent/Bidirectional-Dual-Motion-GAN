## [Bidirectional-Dual-Motion-GAN](https://github.com/xinyuestudent/Bidirectional-Dual-Motion-GAN)

Official pytorch implementation of “Bidirectional Dual Motion GAN for Video Prediction in Intelligent Driving”
[![docs](https://img.shields.io/badge/docs-latest-blue)](README.md)

![image-20240829182410929](C:/Users/wangx/AppData/Roaming/Typora/typora-user-images/image-20240829182410929.png)

![image-20240829183114139](C:/Users/wangx/AppData/Roaming/Typora/typora-user-images/image-20240829183114139.png)

### **Installation**

```python
# pip install required packages
conda create -n vpred -y python=3.7
conda activate vpred
pip install torch===1.10.2  torchvision torchaudio

# get flownet2-pytorch source
git clone https://github.com/NVIDIA/flownet2-pytorch.git
cd flownet2-pytorch

# install custom layers
bash install.sh

#get depth-pytorch source
git clone git@github.com:CompVis/depth-fm.git
cd depth-fm
wget https://ommer-lab.com/files/depthfm/depthfm-v1.ckpt -P checkpoints/
pip install -r requirements.txt
```

### **Training**

```python
python train.py
```

### **Testing**

```python
python test.py
```

