from vgg16  import VGG
import tensorflow as tf 
import argparse
from SVRNet import SVRNet
import skimage.io as io
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.data.experimental import AUTOTUNE
import numpy as np


def read_data(DATA_DIR,IMG_SIZE):

    data = image_dataset_from_directory(
                    DATA_DIR,
                    batch_size=1,
                    validation_split=None,
                    labels="inferred",
                    label_mode="categorical",
                    color_mode="grayscale",
                    image_size=(IMG_SIZE, IMG_SIZE),                    
                    subset=None,
                    seed=1
                )

    return data 


def main():
    parser = argparse.ArgumentParser(description='classifiction using cnn models')
    parser.add_argument(
       '-p_data','--path_data', required=False, default='Dataset/dataset', type=str, help='path data')
    parser.add_argument(
        '-l', '--length', required=False, default=224, type=int, help='length of image')
    parser.add_argument(
        '-w', '--width', required=False, default=224, type=int, help='width of image')
    parser.add_argument(
        '-m', '--model_name',choices=['VGG16', 'SVRNet'], required=False, default='VGG16', type=str, help='model_name')
    parser.add_argument(
        '-t', '--problem_type', choices=['Classification', 'Regression'], required=False, default='Classification', type=str, help='problem_type')
    parser.add_argument(
        '-o', '--output_nums', required=False, default=8, type=int, help='output_num')
    parser.add_argument(
        '-mw', '--model_width', required=False, default=8, type=int, help='width of the Initial Layer, subsequent layers start from here')
    parser.add_argument(
        '-c', '--num_channel', required=False, default=1, type=int, help='number of channel of the image')
    parser.add_argument(
          '-b', '--batch_size', required=False, default=1, type=int, help='batch size')
 
    args = parser.parse_args()

    num_folds = 5
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=num_folds, shuffle=True)
    path_data = args.path_data
    length = args.length  
    width = args.width
    model_name = args.model_name
    model_width = args.model_width
    num_channel = args.num_channel
    problem_type = args.problem_type
    output_nums = args.output_nums      
    def process(image,label):
        image = tf.cast(image/255. ,tf.float32)
        return image,label

    ds = read_data(path_data, length)
    data = ds.map(process)
    print(ds)
   
    BATCH_SIZE = args.batch_size
       
    inputs= np.concatenate(list(data.map(lambda x, y:x)))
    targets = np.concatenate(list(data.map(lambda x, y:y)))  
   
    # K-fold Cross Validation model evaluation
    fold_no = 1
    # output = [(x,y) for x, y in data]  
    # inputs, targets = zip(*output)
    acc_per_fold = []
    loss_per_fold = []
    for train, test in kfold.split(inputs, targets):          
        if model_name == 'VGG16':
            if os.path.isdir(f'{model_name}_{fold_no}_model_dataset'):
                model = keras.models.load_model(f'{model_name}_{fold_no}_model_dataset')
                EPOCHS = 1           
            else:
                EPOCHS = 100
                model = VGG(length, width, num_channel, model_width, problem_type=problem_type, output_nums=output_nums, dropout_rate=0.3).VGG16_v2()          
       
        elif model_name == 'SVRNet':          
            if os.path.isdir(f'{model_name}_{fold_no}_model_dataset'):
                model = keras.models.load_model(f'{model_name}_{fold_no}_model_dataset')
                EPOCHS = 1           
            else:
                EPOCHS = 100
                model = SVRNet(length, width, num_channel, model_width, problem_type=problem_type, output_nums=output_nums, pooling='max', dropout_rate=0.5).SVRNet()
       
        model.summary()


       
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[tf.keras.metrics.CategoricalAccuracy()])
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
        history = model.fit(inputs[train], targets[train],  
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,  
            shuffle=True, verbose=1,
            callbacks=[callback],
            validation_data=(inputs[test], targets[test])
            )

        # Generate a print
        # print('------------------------------------------------------------------------')
        # print(f'Training for fold {fold_no} ...')        
        print(history.history.keys())

        if not os.path.isdir(f'{model_name}_{fold_no}_model_dataset'):
            model.save(f'{model_name}_{fold_no}_model_dataset')

        loss, acc = model.evaluate(inputs[test], targets[test], verbose=0)

               
        # summarize history for accuracy
        plot1 = plt.figure(1)
        plt.plot(history.history['categorical_accuracy'])
        plt.plot(history.history['val_categorical_accuracy'])
        plt.title(f'{model_name}_{fold_no}_model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(f'{model_name}_{fold_no}_accuracy.png')

        # # summarize history for loss
        plot2 = plt.figure(2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'{model_name}_{fold_no}_model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(f'{model_name}_{fold_no}_model_lost.png')
        # print(f"Score for fold {fold_no}: Loss {loss}, Accuracy {acc*100}%")
        fold_no = fold_no + 1
        acc_per_fold.append(acc * 100)
        loss_per_fold.append(loss)
       
       
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
        print('------------------------------------------------------------------------')
        print('Average scores for all folds:')
        print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
        print(f'> Loss: {np.mean(loss_per_fold)}')
        print('------------------------------------------------------------------------')


       

if __name__ == "__main__":
    main()
