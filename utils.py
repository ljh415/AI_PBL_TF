import json
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.class_weight import compute_class_weight

AUTOTUNE = tf.data.experimental.AUTOTUNE

def open_json(path):
    with open(path, 'r') as f:
        js = json.load(f)
    return js

def convert_bbox_info(bbox_info, return_str=True):
    """
    {'x': 441, 'y': 401, 'width': 435, 'height': 431}
    """
    bbox = [bbox_info['y'],
            bbox_info['x'],
            bbox_info['height'], 
            bbox_info['width']]
    
    if return_str:
        bbox = list(map(str, bbox))

    return bbox

def generate_class_weights(class_series, multi_class=True, one_hot_encoded=False):
    if multi_class:
        # If class is one hot encoded, transform to categorical labels to use compute_class_weight   
        if one_hot_encoded:
            class_series = np.argmax(class_series, axis=1)

        # Compute class weights with sklearn method
        class_labels = np.unique(class_series)
        class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=class_series)
        return dict(zip(class_labels, class_weights))
    else:
        # It is neccessary that the multi-label values are one-hot encoded
        mlb = None
        if not one_hot_encoded:
            mlb = MultiLabelBinarizer()
            class_series = mlb.fit_transform(class_series)

        n_samples = len(class_series)
        n_classes = len(class_series[0])

        # Count each class frequency
        class_count = [0] * n_classes
        for classes in class_series:
            for index in range(n_classes):
                if classes[index] != 0:
                    class_count[index] += 1

        # Compute class weights using balanced method
        class_weights = [n_samples / (n_classes * freq) if freq > 0 else 1 for freq in class_count]
        class_labels = range(len(class_weights)) if mlb is None else mlb.classes_
        return dict(zip(class_labels, class_weights))