import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
from tensorflow.keras.models import Model

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:  # gpu가 있다면, 용량 한도를 5GB로 설정
  tf.config.experimental.set_virtual_device_configuration(gpus[0], 
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9*1024)])

class PetModel(Model):
    def __init__(self, input_shape, classes):
        super().__init__()
        self.INPUT_SHAPE = input_shape
        self.classes = classes
        
    def get_model(self, weights=None, trainable_base=False):
        base_model = tf.keras.applications.EfficientNetB0(input_shape=self.INPUT_SHAPE,
                                                                include_top=False,
                                                                weights='imagenet' if weights is not None else weights,
                                                                classes=self.classes)
        gap_layer = tf.keras.layers.GlobalAveragePooling2D()
        bn_0 = tf.keras.layers.BatchNormalization()
        # dense_layer = tf.keras.layers.Dense(512, activation='relu')
        # bn_1 = tf.keras.layers.BatchNormalization()
        prediction_layer = tf.keras.layers.Dense(self.classes, activation='softmax')
        
        base_model.trainable = trainable_base

        self.model = tf.keras.Sequential([
            base_model,
            gap_layer,
            bn_0,
            # dense_layer,
            # bn_1,
              prediction_layer
        ])
        
        return self.model