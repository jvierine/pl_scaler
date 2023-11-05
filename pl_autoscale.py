import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import h5py
import numpy as n
import matplotlib.pyplot as plt
import glob
import imageio
import re
import os
import scipy.interpolate as sint
import pl_scaler as pld
# force CPU evaluation to allow training and scaling simultaneously
#tf.config.set_visible_devices([],'GPU')

label_model=tf.keras.models.load_model("model_000/label")
scale_model=tf.keras.models.load_model("model_000/scale")


dirname="."



def autoscale(img):
    n_imgs_per_batch=1
    pars=n.zeros([n_imgs_per_batch,2],dtype=n.float32)

    imgst=tf.reshape(img,[1,258,300,1])
#    plt.pcolormesh(imgst[0,:,:,0])
 #   plt.colorbar()
  #  plt.show()
    label_pr=label_model.predict(imgst)
    scale_pr=scale_model.predict(imgst)
    return(label_pr,scale_pr)


def label_files():
    fl = glob.glob("/media/j/f194163e-0385-47e9-8892-182fe5b10ae5/pl/plasma_line/integrated_plasma_line_metadata_hires/*/pl*.h5")
    n.random.shuffle(fl)
#    fl.sort()
    for f in fl:
        h=h5py.File(f,"r")
        print(f)
        for k in h.keys():

            t0_this=h[k]["t1"][()]
            ranges=h[k]["ranges"][()]
            freqs=h[k]["freqs"][()]
            spec=h[k]["spec"][()]

            
            
            ranges_d, freqs_d, spec_d=pld.normalize_pl(ranges, freqs, spec)

            
            
            if len(ranges_d) == 258 and len(freqs_d) == 300:
                ridx=n.arange(len(ranges_d))
                fidx=n.arange(len(freqs_d))
                ridx[0]=-1e3
                ridx[-1]=ridx[-1]+1e3
                fidx[0]=-1e3
                fidx[-1]=fidx[-1]+1e3
                rangefun=sint.interp1d(ridx,ranges_d)
                freqfun=sint.interp1d(fidx,freqs_d)            
                
                spec_d[spec_d<0]=0
                spec_d[spec_d>0.05]=0.05
                spec_d=spec_d/0.05
                label,reg=autoscale(spec_d)
                plt.pcolormesh(freqs_d,ranges_d,spec_d,cmap="plasma")
                
                plt.title(label[0,0])
                if label[0,0]>0.96:
                    plt.axvline(freqfun(reg[0,0]))
                    plt.axhline(rangefun(reg[0,1]))
                plt.xlabel("Doppler shift (MHz)")
                plt.ylabel("Range (km)")                
                plt.tight_layout()
                plt.savefig("cnn-%d.png"%(t0_this))
                plt.close("all")

                    
            else:
                print("low res plasma-line. ignoring")


        h.close()


        

label_files()    
