import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.keras import utils
import h5py
import numpy as n
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2

import pl_data as pld

import time
import traceback
import os
import sys
from tensorflow.python.keras.callbacks import TensorBoard

data_dir="."

def teach_network(n_type="label",
                  bs=64,
                  random_seed=1,
                  n_epochs=2,
                  N=10):

    # scalings:
    # fof2, fe, h'mf, h'e
    if n_type == "label":
        dataset=pld.get_pl_data(dirname=data_dir,N=N,shift=True,bs=bs,fr0=0.0,fr1=0.8,data_type="label", random_seed=random_seed)
        validation_dataset=pld.get_pl_data(dirname=data_dir,N=N,shift=True,bs=bs,fr0=0.8,fr1=1.0,data_type="label", random_seed=random_seed)
        
    if n_type == "scale":
        # find ionograms with an f-region
        dataset=pld.get_pl_data(dirname=data_dir,N=N,shift=True,bs=bs,fr0=0.0,fr1=0.8,data_type="scale" , random_seed=random_seed)
        validation_dataset=pld.get_pl_data(dirname=data_dir,N=N,shift=True,bs=bs,fr0=0.8,fr1=1.0,data_type="scale" , random_seed=random_seed)

    if True:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), strides=(1,1), activation='relu', input_shape=(258, 300,1)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
#            tf.keras.layers.Dropout(0.01), # 0 = no dropouts 1 = all drops out            
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
 #           tf.keras.layers.Dropout(dropout), 
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024,activation="relu"),
#            tf.keras.layers.Dropout(dropout),             
            tf.keras.layers.Dense(1024,activation="relu"),
#            tf.keras.layers.Dropout(0.5)
        ])
        
        if n_type == "label":
            model.add(tf.keras.layers.Dense(1,activation="sigmoid"))
        elif n_type == "scale":
            model.add(tf.keras.layers.Dense(2))
        else:
            print("n_type not recognized. exiting")
            exit(0)

    if n_type == "label":
        model.compile(loss="binary_crossentropy",
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005))
    else:
        model.compile(loss="mse",
                      optimizer=tf.keras.optimizers.Adam())
        
    model.summary()

    # directory name
    model_fname="model_%03d/%s"%(random_seed,n_type)
    # remove any existing folder (playing with fire here...)
    os.system("rm -Rf %s"%(model_fname))
    monitor="val_loss"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_fname,
                                                          monitor=monitor,
                                                          save_best_only=True)
    history = model.fit(dataset,
                        batch_size=bs,
                        validation_data=validation_dataset,
                        epochs=n_epochs,
                        callbacks=[model_checkpoint])


random_seed=0
if len(sys.argv) > 1:
    random_seed = int(sys.argv[1])

print("random seed %d"%(random_seed))

try:
    teach_network(n_type="label", bs=32, n_epochs=10, N=10, random_seed=random_seed)
except:
    traceback.print_exc()
    print("caught exception. exiting.")


try:
    teach_network(n_type="scale", bs=32, n_epochs=10, N=10, random_seed=random_seed)
except:
    traceback.print_exc()    
    print("caught exception. exiting.")


