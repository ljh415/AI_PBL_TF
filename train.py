import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import datetime
import tensorflow as tf

from dataset import Dataset
from model import PetModel
from utils import generate_class_weights

def train(args, model, train_ds, valid_ds, data_size, class_weight=None):

    today = datetime.date.today()
    today = today.strftime('%m_%d')
    
    model_checkpoint_dir_path = f"/media/jaeho/SSD/ai_pbl/checkpoints/{today}/"
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    mc = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_checkpoint_dir_path+"{epoch}-{val_loss:.2f}-{val_accuracy:.2f}.h5",
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    model.compile(
        optimizer = tf.keras.optimizers.Adam(args.learning_rate),
        loss = tf.keras.losses.categorical_crossentropy,
        metrics = ['accuracy']
    )
    if class_weight:
        hist = model.fit(
            train_ds,
            epochs=args.epochs,
            validation_data = valid_ds,
            steps_per_epoch = data_size//args.batch_size,
            callbacks = [es, mc],
            class_weight = class_weight
        )
    else :
        hist = model.fit(
            train_ds,
            epochs=args.epochs,
            validation_data = valid_ds,
            steps_per_epoch = data_size//args.batch_size,
            callbacks = [es, mc],
        )
    
    return hist

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str)
    parser.add_argument("--valid", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--train_base", action='store_true')
    args = parser.parse_args()

    INPUT_SHAPE = (args.resolution, args.resolution, 3)
    train_ds, valid_ds, classes, data_size, class_weight = Dataset(args.batch_size, args.train, args.valid).get_dataset(class_weight=True)
    
    model = PetModel(INPUT_SHAPE, classes).get_model(weights="imagenet", trainable_base=args.train_base)
    hist = train(args, model, train_ds, valid_ds, data_size, class_weight)
