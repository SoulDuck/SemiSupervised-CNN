import cnn
import tensorflow
import numpy as np
import cam
import aug
import tensorflow as tf


def dropout(_input, is_training, keep_prob=0.8):
    if keep_prob < 1:
        output = tf.cond(is_training, lambda: tf.nn.dropout(_input, keep_prob), lambda: _input)
    else:
        output = _input
    return output


def batch_norm(_input, is_training):
    output = tf.contrib.layers.batch_norm(_input, scale=True, \
                                          is_training=is_training, updates_collections=None)
    return output


def weight_variable_msra(shape, name):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.variance_scaling_initializer())


def weight_variable_xavier(shape, name):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def bias_variable(shape, name='bias'):
    initial = tf.constant(0.0, shape=shape)
    return tf.get_variable(name, initializer=initial)


def conv2d_with_bias(_input, out_feature, kernel_size, strides, padding):
    in_feature = int(_input.get_shape()[-1])
    kernel = weight_variable_msra([kernel_size, kernel_size, in_feature, out_feature], name='kernel')
    layer = tf.nn.conv2d(_input, kernel, strides, padding) + bias_variable(shape=[out_feature])
    layer = tf.nn.relu(layer)
    print layer
    return layer


def fc_with_bias(_input, out_features):
    in_fearues = int(_input.get_shape()[-1])
    kernel = weight_variable_xavier([in_fearues, out_features], name='kernel')
    layer = tf.matmul(_input, kernel) + bias_variable(shape=[out_features])
    print layer
    return layer


def avg_pool(_input, k):
    ksize = [1, k, k, 1]
    strides = [1, k, k, 1]
    padding = 'VALID'
    output = tf.nn.avg_pool(_input, ksize, strides, padding)
    return output


def fc_layer(_input, out_feature, act_func='relu', dropout='True'):
    assert len(_input.get_shape()) == 2, len(_input.get_shape())
    in_features = _input.get_shape()[-1]
    w = weight_variable_xavier([in_features, out_feature], name='W')
    b = bias_variable(shape=out_feature)
    layer = tf.matmul(_input, w) + b
    if act_func == 'relu':
        layer = tf.nn.relu(layer)
    return layer


def fc_layer_to_clssses(_input, n_classes):
    in_feature = int(_input.get_shape()[-1])
    W = weight_variable_xavier([in_feature, n_classes], name='W')
    bias = bias_variable([n_classes])
    logits = tf.matmul(_input, W) + bias
    return logits


def build_graph(x_, y_, cam_ind, is_training, aug_flag, actmap_flag, model, random_crop_resize, bn):
    ##### define conv connected layer #######
    n_classes = int(y_.get_shape()[-1])
    image_size = int(x_.get_shape()[-2])
    if model == 'vgg_11':
        print 'Model : {}'.format('vgg 11')
        conv_out_features = [64, 128, 256, 256, 512, 512, 512, 512]
        conv_kernel_sizes = [3, 3, 3, 3, 3, 3, 3, 3]
        conv_strides = [1, 1, 1, 1, 1, 1, 1, 1]
        before_act_bn_mode = [False, False, False, False, False, False, False, False, False]
        after_act_bn_mode = [False, False, False, False, False, False, False, False, False]
        if bn == True:
            before_act_bn_mode = [True, True, True, True, True, True, True, True]
        allow_max_pool_indices = [0, 1, 2, 3, 5, 7]

    if model == 'vgg_13':
        conv_out_features = [64, 64, 128, 128, 256, 256, 512, 512, 512, 512]
        conv_kernel_sizes = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        conv_strides = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        before_act_bn_mode = [False, False, False, False, False, False, False, False, False, False]
        after_act_bn_mode = [False, False, False, False, False, False, False, False, False, False]
        if bn == True:
            before_act_bn_mode = [True, True, True, True, True, True, True, True, True, True]
        allow_max_pool_indices = [1, 3, 5, 7, 9]

    if model == 'vgg_16':
        conv_out_features = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
        conv_kernel_sizes = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        conv_strides = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        before_act_bn_mode = [False, False, False, False, False, False, False, False, False, False, False, False, False]
        after_act_bn_mode = [False, False, False, False, False, False, False, False, False, False, False, False, False]
        if bn == True:
            before_act_bn_mode = [True, True, True, True, True, True, True, True, True, True, True, True, True]

        allow_max_pool_indices = [1, 3, 6, 9, 12]

    if model == 'vgg_19':
        conv_out_features = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
        conv_kernel_sizes = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        conv_strides = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        before_act_bn_mode = [False, False, False, False, False, False, False, False, False, False, False, False, False,
                              False, False]
        after_act_bn_mode = [False, False, False, False, False, False, False, False, False, False, False, False, False,
                             False, False]
        if bn == True:
            before_act_bn_mode = [True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                                  True, True]
        allow_max_pool_indices = [1, 3, 7, 9, 11, 15]

    ###VGG Paper ###
    """
    VGG-11 64 max 128 max 256 256 max 512 512 max 512 512 max 4096 4096 1000  
    VGG-11 64 LRN max 128 max 256 256 max 512 512 max 512 512 max 4096 4096 1000
    VGG-13 64 64 LRN max 128 128 max 256 256 max 512 512 max 512 512 max 4096 4096 1000
    VGG-16 64 64 LRN max 128 128 max 256 256 256 max 512 512 512 max 512 512 512 max 4096 4096 1000
    VGG-16 64 64 LRN max 128 128 max 256 256 256 max 512 512 512 max 512 512 512 max 4096 4096 1000
    VGG-16 64 64 LRN max 128 128 max 256 256 256 256 max 512 512 512 512 max 512 512 512 512 max 4096 4096 1000

    """

    if aug_flag:
        print 'aug : True'
        if random_crop_resize is None:
            random_crop_resize = int(x_.get_shape()[-2])

        x_ = tf.map_fn(lambda image: aug.aug_lv0(image, is_training, image_size=random_crop_resize), x_)
        x_ = tf.identity(x_, name='aug_')
    assert len(conv_out_features) == len(conv_kernel_sizes) == len(conv_strides), \
        '{}{}{}'.format(len(conv_out_features), len(conv_kernel_sizes), len(conv_strides))
    layer = x_
    for i in range(len(conv_out_features)):
        with tf.variable_scope('conv_{}'.format(str(i))) as scope:
            if before_act_bn_mode[i] == True:
                layer = batch_norm(layer, is_training)

            layer = conv2d_with_bias(layer, conv_out_features[i], kernel_size=conv_kernel_sizes[i], \
                                     strides=[1, conv_strides[i], conv_strides[i], 1], padding='SAME')
            if i in allow_max_pool_indices:
                print 'max pooling layer : {}'.format(i)
                layer = tf.nn.max_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                print layer
            layer = tf.nn.relu(layer)
            if after_act_bn_mode[i] == True:
                layer = batch_norm(layer, is_training)
            # layer=tf.nn.dropout(layer , keep_prob=conv_keep_prob)
            layer = tf.cond(is_training, lambda: tf.nn.dropout(layer, keep_prob=1.0), lambda: layer)
    end_conv_layer = tf.identity(layer, name='top_conv')
    layer = tf.contrib.layers.flatten(end_conv_layer)
    print "num of Classes : ", n_classes
    logits_gap = cnn.gap('gap', end_conv_layer, n_classes)
    cam_ = cam.get_class_map('gap', end_conv_layer, cam_ind, image_size)

    ##### define fully connected layer #######
    fc_out_features = [1024, 1024]
    before_act_bn_mode = [False, False]
    after_act_bn_mode = [False, False]
    for i in range(len(fc_out_features)):
        with tf.variable_scope('fc_{}'.format(str(i))) as scope:
            if before_act_bn_mode[i] == True:
                print 'batch normalization {}'.format(i)
                layer = batch_norm(layer, is_training)
            layer = fc_with_bias(layer, fc_out_features[i])
            layer = tf.nn.relu(layer)
            layer = tf.cond(is_training, lambda: tf.nn.dropout(layer, keep_prob=0.5), lambda: layer)
            if after_act_bn_mode[i] == True:
                layer = batch_norm(layer, is_training)
    print n_classes
    logits_fc = fc_layer_to_clssses(layer, n_classes)
    if actmap_flag:
        print "logits from Global Average Pooling , No Fully Connected layer "
        logits = logits_gap
    else:
        print "logits from fully connected layer "
        logits = logits_fc

    logits = tf.identity(logits, name='logits')
    print "logits's shape : {}".format(logits)
    return logits


def train_algorithm_momentum(logits, labels, learning_rate, use_nesterov, l2_loss):
    print 'Optimizer : Momentum'
    print 'Use Nesterov : ', use_nesterov
    print 'L2 Loss : ', l2_loss
    prediction = tf.nn.softmax(logits, name='softmax')
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels),
                                   name='cross_entropy')

    momentum = 0.9;
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=momentum, use_nesterov=use_nesterov)
    if l2_loss:
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()], name='l2_loss')
        weight_decay = 1e-4
        train_op = optimizer.minimize(cross_entropy + l2_loss * weight_decay, name='train_op')
    else:
        train_op = optimizer.minimize(cross_entropy, name='train_op')
    correct_prediction = tf.equal(
        tf.argmax(prediction, 1),
        tf.argmax(labels, 1), name='correct_prediction')

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32), name='accuracy')
    return train_op, accuracy, cross_entropy, prediction


def train_algorithm_adam(logits, labels, learning_rate, l2_loss):
    prediction = tf.nn.softmax(logits, name='softmax')
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels),
                                   name='cross_entropy')
    optimizer = tf.train.AdamOptimizer(learning_rate)
    if l2_loss:
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()], name='l2_loss')
        weight_decay = 1e-4
        train_op = optimizer.minimize(cross_entropy + l2_loss * weight_decay, name='train_op')
    else:
        train_op = optimizer.minimize(cross_entropy, name='train_op')
    correct_prediction = tf.equal(
        tf.argmax(prediction, 1),
        tf.argmax(labels, 1), name='correct_prediction')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32), name='accuracy')
    return train_op, accuracy, cross_entropy, prediction


def train_algorithm_grad(logits, labels, learning_rate, l2_loss):
    prediction = tf.nn.softmax(logits, name='softmax')
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels),
                                   name='cross_entropy')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    if l2_loss:
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()], name='l2_loss')
        weight_decay = 1e-4
        train_op = optimizer.minimize(cross_entropy + l2_loss * weight_decay, name='train_op')
    else:
        train_op = optimizer.minimize(cross_entropy, name='train_op')
    correct_prediction = tf.equal(
        tf.argmax(prediction, 1),
        tf.argmax(labels, 1), name='correct_prediction')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32), name='accuracy')
    return train_op, accuracy, cross_entropy, prediction


def define_inputs(shape, n_classes):
    images = tf.placeholder(tf.float32, shape=shape, name='x_')
    labels = tf.placeholder(tf.float32, shape=[None, n_classes], name='y_')
    cam_ind = tf.placeholder(tf.int32, shape=[], name='cam_ind')
    learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    return images, labels, cam_ind, learning_rate, is_training


def sess_start(logs_path):
    saver = tf.train.Saver(max_to_keep=10000000)
    sess = tf.Session()
    summary_writer = tf.summary.FileWriter(logs_path)
    summary_writer.add_graph(tf.get_default_graph())
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)
    return sess, saver, summary_writer


def write_acc_loss(summary_writer, prefix, loss, acc, step):
    summary = tf.Summary(value=[tf.Summary.Value(tag='loss_{}'.format(prefix), simple_value=float(loss)),
                                tf.Summary.Value(tag='accuracy_{}'.format(prefix), simple_value=float(acc))])
    summary_writer.add_summary(summary, step)


if __name__ == '__main__':
    x_, y_, lr_, is_training = define_inputs(shape=[None, 299, 299, 3], n_classes=2)
    build_graph(x_=x_, y_=y_, is_training=is_training, aug_flag=True, actmap_flag=True, model='vgg_11',
                random_crop_resize=224)



