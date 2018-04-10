import random
import numpy as np
import cifar
import glob
def next_batch(batch_size , imgs, labs, fnames=None):
    indices = random.sample(range(np.shape(labs)[0]), batch_size)
    if not type(imgs).__module__ == np.__name__:  # check images type to numpy
        imgs = np.asarray(imgs)
    imgs = np.asarray(imgs)
    batch_xs = imgs[indices]
    batch_ys = labs[indices]
    if not fnames is None:
        batch_fs = fnames[indices]
    else:
        batch_fs = None
    return batch_xs, batch_ys , batch_fs




def cls2onehot(cls, depth):
    debug_flag=False
    if not type(cls).__module__ == np.__name__:
        cls=np.asarray(cls)
    cls=cls.astype(np.int32)
    debug_flag = False
    labels = np.zeros([len(cls), depth] , dtype=np.int32)
    for i, ind in enumerate(cls):
        labels[i][ind:ind + 1] = 1
    if __debug__ == debug_flag:
        print '#### data.py | cls2onehot() ####'
        print 'show sample cls and converted labels'
        print cls[:10]
        print labels[:10]
        print cls[-10:]
        print labels[-10:]
    return labels


def cifar_input(onehot=True):
    train_filenames=glob.glob('./cifar_10/cifar-10-batches-py/data_batch*')
    test_filenames=glob.glob('./cifar_10/cifar-10-batches-py/test_batch*')
    train_imgs , train_labs = cifar.get_images_labels(*train_filenames)
    test_imgs, test_labs = cifar.get_images_labels(*test_filenames)
    if onehot :
        train_labs=cls2onehot(train_labs , 10)
        test_labs = cls2onehot(test_labs, 10)
    print 'train imgs shape : {}'.format(np.shape(train_imgs))
    print 'train labs shape : {}'.format(np.shape(train_labs))
    print 'test imgs shape : {}'.format(np.shape(test_imgs))
    print 'test labs shape : {}'.format(np.shape(test_labs))
    return train_imgs , train_labs , test_imgs , test_labs







