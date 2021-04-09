# BoltAngleDetection
This application is an implementation of bolt angle detection by FasterRcnn combined with a serial of computer vision processing. The application generates bounding boxes and measures the angle of bolts automatically by taking one photo of bolts

### Environment
```
conda create -n BoltAngleDetection pip python=3.6
activate BoltAngleDetection
conda install tensorflow-gpu=1.15.0
conda install -c anaconda protobuf
pip install pillow
pip install lxml
pip install jupyter
pip install matplotlib
pip install pandas
pip install opencv-python
pip install cython
```

### To train the model:
