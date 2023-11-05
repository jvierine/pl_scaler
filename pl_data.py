import tensorflow as tf
import numpy as n
import matplotlib.pyplot as plt
import glob
import os
import imageio
import h5py

class random_shift_data(tf.keras.utils.Sequence):
    """
    generate data on the fly with random x shifts
    """
    def __init__(self,
                 imgs,
                 scaling,
                 batch_size=64,
                 N=10,
                 x_width=10,
                 y_width=10,
                 shift=True,
                 fr0=0,
                 fr1=1.0,
                 data_type="label"):
        
        idx=n.arange(imgs.shape[0])
        imgs0=imgs[int(fr0*len(idx)):int(fr1*len(idx)),:,:]
        self.imgs=imgs0
        self.scaling=scaling[int(fr0*len(idx)):int(fr1*len(idx)),:]
        self.batch_size=batch_size
        self.N=N
        self.x_width=x_width
        self.y_width=y_width        
        self.n_im=self.imgs.shape[0]
        self.shift=shift
        self.data_type=data_type
        if data_type not in ["scale","label"]:
            print("data type should be one of scale,label")
            exit(0)
        
    def __len__(self):
        return(int(self.n_im*self.N/float(self.batch_size)))
    
    def __getitem__(self,idx):
        imi = n.array(n.mod(self.batch_size*idx + n.arange(self.batch_size,dtype=n.int64),self.n_im),dtype=n.int64)
        img_out=n.zeros([self.batch_size,self.imgs.shape[1],self.imgs.shape[2]],dtype=n.float32)
        scale_out=n.zeros([self.batch_size,self.scaling.shape[1]],dtype=n.float32)
        
        for i in range(self.batch_size):
            # randomly pick batch_size images from dataset
            # shift them randomly to create more data
            im0=n.copy(self.imgs[imi[i],:,:])
            xi=0.0
            yi=0.0
            if self.shift:
                # how many pixels do we shift frequency
                xi=int(n.random.rand(1)*self.x_width-self.x_width/2.0)
                # how many pixels do we shift range
                yi=int(n.random.rand(1)*self.y_width-self.y_width/2.0)
                
                # if frequency shift wraps around, then don't shift
                # otherwise the fof2 would occur at high frequencies
                # the first parameter is either fof2 or fe, depending on mode
                if self.data_type!="label":
                    if self.scaling[imi[i],0]+xi < 0:
                        xi=0
                    if self.scaling[imi[i],1]+yi < 0:
                        yi=0
                    
                im0=n.roll(im0,xi,axis=1)
                im0=n.roll(im0,yi,axis=0)                
            img_out[i,:,:]=im0
            
            # if the data is used for a regression task, shift regression values
            if self.data_type == "scale":
                # first parameter is frequency
                scale_out[i,0]=self.scaling[imi[i],0]+xi
                # the second parameter is height
                scale_out[i,1]=self.scaling[imi[i],1]+yi

                if False:
                    plt.pcolormesh(im0)
                    plt.axvline(scale_out[i,0])
                    plt.axhline(scale_out[i,1])
                    plt.colorbar()
                    plt.show()
                
            else: # label
                # shifting doesn't affect labeling, so keep labels as is
                scale_out[i,:]=self.scaling[imi[i],:]
                    
        img_out.shape=(img_out.shape[0],img_out.shape[1],img_out.shape[2],1)
        return(img_out,scale_out)

    
def get_images(dirname=".",data_type="label",random_seed=42):
    """
    data type: "scale" or "label"
    """
    fl=glob.glob("%s/1*.h5"%(dirname))
    fl.sort()
    # fixed random seed, so that we always use the
    # same images for validation and learning (otherwise there will be overlap)
    n.random.seed(random_seed)
    n.random.shuffle(fl)
    imgs=[]
    scalings=[]
    for f in fl:
        h=h5py.File(f,"r")
        img=h["spec"][()]
        img[img<0]=0
        img[img>0.05]=0.05
        img=img/0.05
        if data_type == "label":
            s=[0]
            if n.isnan(h["x"][()]) or n.isnan(h["y"][()]):
                s[0]=0
            else:
                s[0]=1
            scalings.append(s)
            imgs.append(img)
        else:
            if n.isnan(h["x"][()]) or n.isnan(h["y"][()]):
                pass
            else:
                s=[h["x"][()],h["y"][()]]
                scalings.append(s)
                imgs.append(img)
        h.close()
    return(n.array(imgs),n.array(scalings))

def get_pl_data(dirname=".",
                bs=64,
                N=10,
                x_width=50,
                shift=True,
                random_seed=42,
                fr0=0.0,fr1=1.0,
                data_type="label"):

    # get collection of all images and scaling parameters
    im,sc=get_images(dirname=dirname,data_type=data_type, random_seed=random_seed)    
    n_images=im.shape[0]
    print("Total number of images %d"%(n_images))
    return(random_shift_data(im,
                             sc,
                             batch_size=bs,
                             N=N,
                             x_width=x_width,shift=shift,fr0=fr0,fr1=fr1,data_type=data_type))

if __name__ == "__main__":

    pld=get_pl_data(data_type="label")
    a,b=pld[0]
    for i in range(4):
        plt.pcolormesh(a[i,:,:,0])
        plt.title(b[i,0])
        plt.show()


    pld=get_pl_data(data_type="scale")
    a,b=pld[0]
    for i in range(32):
        plt.pcolormesh(a[i,:,:,0])
        plt.axvline(b[i,0])
        plt.axhline(b[i,1])        
        plt.show()
        
    print(a.shape)
    print(b.shape)    
