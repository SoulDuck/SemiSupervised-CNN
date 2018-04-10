import tensorflow as tf
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
def clahe_equalized(img):
    if len(img.shape) == 2:
        img=np.reshape(img, list(np.shape(img)) +[1])
    assert (len(img.shape)==3)  #4D arrays
    img=img.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if img.shape[-1] ==3: # if color shape
        for i in range(3):
            img[:, :, i]=clahe.apply(np.array(img[:,:,i], dtype=np.uint8))
    elif img.shape[-1] ==1: # if Greys,
        img = clahe.apply(np.array(img, dtype = np.uint8))
    return img


def random_rotate_90(images):
    k=np.random.randint(0,4)
    images=np.rot90(images , k , axes =(1,2))
    return images



def random_rotate_with_PIL(image):

    ### usage: map(random_rotate , images) ###
    image=Image.fromarray(image)
    ind=random.randint(0,180)
    minus = random.randint(0,1)
    minus=bool(minus)
    if minus==True:
        ind=ind*-1
    img = image.rotate(ind)
    if __debug__ == True:
        print ind
    return np.asarray(img)

#==== histogram equalization
def histo_equalized(img):
    assert (len(np.shape(img))==2)  ,' image shape : {} '.format(np.shape(img)) #4D arrays
    return cv2.equalizeHist(np.array(img, dtype = np.uint8))




def aug_lv0(image_ , is_training , image_size):

    def aug_with_train(image, image_size):
        ch=int(image.get_shape()[-1])

        image = tf.image.resize_image_with_crop_or_pad(image, image_size+4, image_size+4)
        image = tf.random_crop(image, [image_size, image_size, ch])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)

        #Brightness / saturatio / constrast provides samll gains 2%~5% on cifar

        image = tf.image.random_brightness(image, max_delta=63. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.8)
        image = tf.image.per_image_standardization(image)
        return image

    def aug_with_test(image , image_size):

        image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
        image = tf.image.per_image_standardization(image)
        return image

    image=tf.cond(is_training , lambda : aug_with_train(image_ , image_size=image_size)  , \
                  lambda  : aug_with_test(image_ , image_size=image_size) )


    return image




if __name__ == '__main__':
    img=Image.open('tmp/abnormal_actmap.png').convert('L')
    fig = plt.figure()
    ax=fig.add_subplot(132)
    HE_img = histo_equalized(img)
    ax.imshow(HE_img)
    ax = fig.add_subplot(133)
    rotated_img=random_rotate_with_PIL(img)
    ax.imshow(rotated_img)
    plt.show()