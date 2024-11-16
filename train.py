import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from object_detection.utils import dataset_util

# Clone TensorFlow Object Detection API
!git clone https://github.com/tensorflow/models.git
%cd models/research
!protoc object_detection/protos/*.proto --python_out=.
!cp object_detection/packages/tf2/setup.py .
!pip install .

# Download a pre-trained model
!mkdir -p training
!wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz
!tar -xvf efficientdet_d0_coco17_tpu-32.tar.gz -C training
