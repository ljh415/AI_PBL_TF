import tensorflow as tf

class PetModel:
    def __init__(self, input_shape, classes):
        self.INPUT_SHAPE = input_shape
        self.classes = classes
    
    def get_model(self, trainable_base=False):
        base_model = tf.keras.applications.EfficientNetB0(input_shape=self.INPUT_SHAPE,
                                                            include_top=False,
                                                            weights='imagenet',
                                                            classes=self.classes)
        gap_layer = tf.keras.layers.GlobalAveragePooling2D()
        bn_0 = tf.keras.layers.BatchNormalization()
        dense_layer = tf.keras.layers.Dense(512, activation='relu')
        bn_1 = tf.keras.layers.BatchNormalization()
        prediction_layer = tf.keras.layers.Dense(self.classes, activation='softmax')
        
        base_model.trainable = trainable_base

        model = tf.keras.Sequential([
            base_model,
            gap_layer,
            bn_0,
            dense_layer,
            bn_1,
            prediction_layer
        ])
        
        return model