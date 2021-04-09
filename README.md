# BoltAngleDetection
This application is an implementation of bolt angle detection by FasterRcnn combined with a serial of computer vision processing. The application generates bounding boxes and measures the angle of bolts automatically by taking one photo of bolts

![image](https://github.com/BingXiong1995/BoltAngleDetection/blob/main/results/%E5%9B%BE%E7%89%871.jpg)

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
Create folder named train and test in images folder and put images with xml labels in it and then generate csc files by using:
```
python xml_to_csv.py
```
You can start trainning your model by using:
```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```
To launch tensorflow board:
```
tensorboard --logdir=training
```
After getting checkpoints in trainning folder you can generate your faster rcnn model by using(change xxx to your check point name):
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
```







