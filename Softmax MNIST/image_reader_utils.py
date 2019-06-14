from PIL import Image
import numpy as np
import os

#img = Image.open("C:\\Users\\Juan.Zarco\\Documents\\GitHub\\Tensorflow-Examples\\Softmax MNIST\\fruits-360\\Training\\Apple Crimson Snow\\0_100.jpg")
#img.load()
#data = np.asarray(img,dtype="int32")
#print(data.shape)
#print(data[:,:,0]/255.0)

def read_images(indir=None,normalize=True):
    image_data = []
    for r,d,f in os.walk(indir):
        for fname in f:
            img = Image.open(os.path.join(r,fname))
            img.load()
            data = np.asarray(img,dtype="int32")
            data= data/255.0
            image_data.append(data)

    return image_data

def build_imgdata_from_dirs(indir=None,classification=True):
    """
    :param indir:
    :param classification:
    :return: dataset with class labels if classification is set true
    """
    total_images = []
    Y = []
    if classification:
        for r,d,f in os.walk(indir):
            print("Root: ",r)
            for i,dir in enumerate(d):
                print("Directory: ",dir)
                class_ = i + 1
                images = read_images(indir=os.path.join(r,dir))
                total_images = total_images + images
                Y = Y + [class_]*len(images)

    X = np.array(total_images)
    Y = np.array(Y)
    return X,Y


def save_image(npdata,outdir,filename,isNormalized=True):
    if isNormalized:
        npdata = npdata*255

    img = Image.fromarray(np.asarray(np.clip(npdata,0,255),dtype="uint8"),"RGB")
    img.save(outdir+filename)

def get_directories(indir=None):
    dir_list = []
    for r,d,_ in os.walk(indir):
        dir_list.append(str(os.path.join(r,str(d))))

    return dir_list
"""
fp = "C:\\Users\\Juan.Zarco\\Documents\\GitHub\\Tensorflow-Examples\\Softmax MNIST\\fruits-360\\Training"

for r,d,f in os.walk(fp):
    print(r)
    print(d)
    for dir in d:
        print(dir)
        print(os.path.join(r,dir))
        break
    break
"""