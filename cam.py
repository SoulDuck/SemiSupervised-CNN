import os
import matplotlib

if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from skimage.io import imsave
import scipy.misc
import sys
from PIL import Image


def get_class_map(name, x, cam_ind, im_width):
    out_ch = int(x.get_shape()[-1])
    conv_resize = tf.image.resize_bilinear(x, [im_width, im_width])
    with tf.variable_scope(name, reuse=True) as scope:
        label_w = tf.gather(tf.transpose(tf.get_variable('w')), cam_ind)
        label_w = tf.reshape(label_w, [-1, out_ch, 1])
    conv_resize = tf.reshape(conv_resize, [-1, im_width * im_width, out_ch])
    classmap = tf.matmul(conv_resize, label_w, name='classmap')
    classmap = tf.reshape(classmap, [-1, im_width, im_width], name='classmap_reshape')
    return classmap


def inspect_cam(sess, cam, top_conv, test_imgs, test_labs, global_step, num_images, x_, y_, phase_train, y, ):
    debug_flag = False
    try:
        os.mkdir('./out');
    except Exception:
        pass;

    for s in range(num_images):
        save_dir = './out/img_{}'.format(s)
        try:
            os.mkdir(save_dir);
        except Exception:
            pass;
        if __debug__ == debug_flag:
            print test_imgs[s].shape
        if test_imgs[s].shape[-1] == 1:
            plt.imsave('{}/image_test.png'.format(save_dir),
                       test_imgs[s].reshape([test_imgs[s].shape[0], test_imgs.shape[1]]))
        else:
            plt.imsave('{}/image_test.png'.format(save_dir), test_imgs[s])
        img = test_imgs[s:s + 1]
        label = test_labs[s:s + 1]
        conv_val, output_val = sess.run([top_conv, y], feed_dict={x_: img, phase_train: False})
        cam_ans = sess.run(cam, feed_dict={y_: label, top_conv: conv_val})
        cam_vis = list(map(lambda x: (x - x.min()) / (x.max() - x.min()), cam_ans))
        for vis, ori in zip(cam_vis, img):
            if ori.shape[-1] == 1:  # grey
                plt.imshow(1 - ori.reshape([ori.shape[0], ori.shape[1]]))
            plt.imshow(vis.reshape([vis.shape[0], vis.shape[1]]), cmap=plt.cm.jet, alpha=0.5, interpolation='nearest',
                       vmin=0, vmax=1)
            cmap_file = '{}/cmap_{}.png'.format(save_dir, global_step)
            plt.savefig(cmap_file)
            plt.close();


def overlay(actmap, ori_img, save_path, factor):
    assert factor <= 1 and factor >= 0
    cmap = plt.cm.jet
    plt.imsave(fname='tmp.png', arr=cmap(actmap))
    cam_img = Image.open('tmp.png')
    # np_cam_img=np.asarray(cam_img).astype('uint8') #img 2 numpy
    ori_img = Image.fromarray(ori_img.astype('uint8')).convert("RGBA")
    overlay_img = Image.blend(ori_img, cam_img, factor).convert('RGB')
    plt.imshow(overlay_img)
    plt.imsave(save_path, overlay_img)
    save_dir, name = os.path.split(save_path)
    name = os.path.splitext(name)[0] + '_ori.png'
    plt.imsave(os.path.join(save_dir, name), ori_img)
    plt.close();
    os.remove('tmp.png')

    return np.asarray(overlay_img)
    """
    if test_imgs.shape[-1] == 1:  # grey
        plt.imshow(1 -img.reshape([test_imgs.shape[1], test_imgs.shape[2]]))
        plt.show()
    """


def eval_inspect_cam(sess, cam, cam_ind, top_conv, test_imgs, x, y_, phase_train, y, save_root_folder):
    num_images = len(test_imgs[:])
    ABNORMAL_LABEL = np.asarray([[0, 1]])
    NORMAL_LABEL = np.asarray([[1, 0]])
    ABNORMAL_CLS = np.argmax(ABNORMAL_LABEL, axis=1)
    NORMAL_CLS = np.argmax(ABNORMAL_LABEL, axis=1)
    try:
        os.mkdir('./out');
    except Exception as e:
        print e
        pass;
    if not os.path.isdir(save_root_folder):
        os.mkdir(save_root_folder)
    for s in range(num_images):
        msg = '\r {}/{}'.format(s, num_images)
        sys.stdout.write(msg)
        sys.stdout.flush()
        save_dir = './{}/img_{}'.format(save_root_folder, s)
        try:
            os.mkdir(save_dir);
        except Exception as e:
            print e;
        if __debug__ == False:
            print 'test imgs shape : ', test_imgs[s].shape

        if test_imgs[s].shape[-1] == 1:
            plt.imsave('{}/image_test.png'.format(save_dir), test_imgs[s].reshape([test_imgs[s].shape[0] \
                                                                                      , test_imgs.shape[1]]))
        else:
            plt.imsave('{}/image_test.png'.format(save_dir), test_imgs[s])
        img = test_imgs[s].reshape(1, test_imgs[s].shape[0], test_imgs[s].shape[1], -1)
        conv_val, output_val = sess.run([top_conv, y], feed_dict={x: img, phase_train: False})

        cam_ans_abnormal = sess.run(cam, feed_dict={y_: ABNORMAL_LABEL, cam_ind: ABNORMAL_CLS[0], top_conv: conv_val,
                                                    phase_train: False})
        cam_ans_normal = sess.run(cam, feed_dict={y_: NORMAL_LABEL, cam_ind: NORMAL_CLS[0], top_conv: conv_val,
                                                  phase_train: False})

        cam_vis_abnormal = list(map(lambda x: (x - x.min()) / (x.max() - x.min()), cam_ans_abnormal))
        cam_vis_normal = list(map(lambda x: (x - x.min()) / (x.max() - x.min()), cam_ans_normal))

        cam_vis_abnormal, cam_vis_normal = map(lambda x: np.squeeze(x), [cam_vis_abnormal, cam_vis_normal])
        cam_vis_abnormal = cam_vis_abnormal.reshape([img.shape[1], img.shape[2]])
        cam_vis_normal = cam_vis_normal.reshape([img.shape[1], img.shape[2]])

        cmap = plt.cm.jet
        # plt.imshow(cam_vis_abnormal, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest', vmin=0, vmax=1)
        cam_vis_abnormal = cmap(cam_vis_abnormal)
        plt.imsave('{}/abnormal_actmap.png'.format(save_dir), cam_vis_abnormal)
        plt.imsave('{}/normal_actmap.png'.format(save_dir), cam_vis_normal)
        """
        print np.shape(cam_vis_abnormal)
        print np.shape(cam_vis_normal)
        for vis , ori in zip(cam_vis_abnormal , img):
            print 'vis shape : ',np.shape(vis)
            print 'ori shape : ',np.shape(ori)
            if ori.shape[-1]==1: #grey

                plt.imshow( 1-ori.reshape([ori.shape[1] , ori.shape[2]]))
            vis_abnormal=vis.reshape([ori.shape[0], ori.shape[1]])
            print 'vis shape ' , np.shape(vis)
            plt.imshow( vis_abnormal, cmap=plt.cm.jet , alpha=0.5 , interpolation='nearest' , vmin=0 , vmax=1)
            plt.imsave('{}/abnormal_actmap.png'.format(save_dir), vis_abnormal)
        for vis, ori in zip(cam_vis_normal, img):
            if ori.shape[-1] == 1:  # grey
                plt.imshow(1 - ori.reshape([ori.shape[0], ori.shape[1]]))
            vis_normal = vis.reshape([vis.shape[0], vis.shape[1]])
            plt.imshow(vis_normal, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest', vmin=0, vmax=1)
            plt.imsave('{}/normal_actmap.png'.format(save_dir), vis_normal)
        """
