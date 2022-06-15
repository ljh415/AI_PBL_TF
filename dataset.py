import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import pandas as pd
import tensorflow as tf

from utils import *

class Dataset:
    def __init__(self, batch_size, train_path, valid_path, test_path=None):
        if 'sample' not in train_path:
            train_flag = True
        else :
            train_flag = False
        self.train_df = pd.read_csv(train_path)
        self.valid_df = pd.read_csv(valid_path)
        self.label_map = {}
        for idx, label in enumerate(self.valid_df['labels'].unique()):
            self.label_map[label] = idx
        self.label_len = len(self.label_map)
        self.DATA_SIZE = len(self.train_df)
        self.BATCH_SIZE = batch_size
        self.prepare(train_flag)

    def prepare(self, train_flag=False):
        # preprocess for valid csv
        self.valid_df['labels'] = self.valid_df['labels'].apply(lambda x: self.label_map[x])
        if train_flag:
            self.train_df['labels'] = self.train_df['labels'].apply(lambda x: self.label_map[x])

        # column sorting
        self.train_df = self.train_df[['image_path', 'bbox', 'labels']]
        self.valid_df = self.valid_df[['image_path', 'bbox', 'labels']]
        self.train_target = self.train_df.pop('labels')
        self.valid_target = self.valid_df.pop('labels')


    def preprocess(self, x, y):
        bbox = tf.strings.split(x[1], ",")
        bbox = tf.strings.to_number(bbox, out_type=tf.int32)
        img = tf.io.read_file(x[0])
        img = tf.image.decode_and_crop_jpeg(img, bbox, channels=0)
        img /= 255
        img = tf.image.resize(img, [224, 224])
        img.set_shape([224, 224, 3])

        label = tf.one_hot(y, self.label_len)

        return img, label
    
    def augment(self, image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        return image, label
    
    def get_dataset(self, class_weight=False):

        # train
        train_ds = tf.data.Dataset.from_tensor_slices((self.train_df.values, self.train_target.values))
        train_ds = train_ds.map(self.preprocess, num_parallel_calls=AUTOTUNE)
        train_ds = train_ds.map(self.augment, num_parallel_calls=AUTOTUNE)
        train_ds = train_ds.batch(self.BATCH_SIZE)
        train_ds = train_ds.shuffle(10)
        train_ds = train_ds.repeat()
        train_ds = train_ds.prefetch(AUTOTUNE)

        # valid
        valid_ds = tf.data.Dataset.from_tensor_slices((self.valid_df.values, self.valid_target.values))
        valid_ds = valid_ds.map(self.preprocess, num_parallel_calls=AUTOTUNE)
        valid_ds = valid_ds.batch(self.BATCH_SIZE)
        valid_ds = valid_ds.prefetch(AUTOTUNE)
        
        if class_weight:
            class_weight = generate_class_weights(self.train_target, multi_class=True, one_hot_encoded=False)
            return train_ds, valid_ds, self.label_len, self.DATA_SIZE, class_weight
        else :
            return train_ds, valid_ds, self.label_len, self.DATA_SIZE
