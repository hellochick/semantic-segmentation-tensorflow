from network import Network
from tools import *
import tensorflow as tf
import os

class FCN8s(Network):
    def __init__(self, is_training=False, num_classes=151, input_size=[384, 384]):
        self.input_size = input_size
    
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
        self.img_tf = preprocess(self.x, self.input_size)
        
        super().__init__({'data': self.img_tf}, num_classes, is_training)

    def setup(self, is_training, num_classes):
        (self.feed('data')
             .zero_padding(paddings=100, name='padding1')
             .conv(3, 3, 64, 1, 1, name='conv1_1')
             .zero_padding(paddings=1, name='padding2')
             .conv(3, 3, 64, 1, 1, name='conv1_2')
             .max_pool(2, 2, 2, 2, name='pool1')
             .zero_padding(paddings=1, name='padding3')
             .conv(3, 3, 128, 1, 1, name='conv2_1')
             .zero_padding(paddings=1, name='padding4')
             .conv(3, 3, 128, 1, 1, name='conv2_2')
             .zero_padding(paddings=1, name='padding5')
             .max_pool(2, 2, 2, 2, name='pool2')
             .zero_padding(paddings=1, name='padding6')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .zero_padding(paddings=1, name='padding7')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .zero_padding(paddings=1, name='padding8')
             .conv(3, 3, 256, 1, 1, name='conv3_3')
             .max_pool(2, 2, 2, 2, name='pool3')
             .scale(0.00001, name='scale_pool3')
             .conv(1, 1, num_classes, 1, 1, name='score_pool3'))

        (self.feed('pool3')
             .zero_padding(paddings=1, name='padding9')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .zero_padding(paddings=1, name='padding10')
             .conv(3, 3, 512, 1, 1, name='conv4_2')
             .zero_padding(paddings=1, name='padding11')
             .conv(3, 3, 512, 1, 1, name='conv4_3')
             .zero_padding(paddings=1, name='padding12')
             .max_pool(2, 2, 2, 2, name='pool4')
             .scale(0.01, name='scale_pool4')
             .conv(1, 1, num_classes, 1, 1, name='score_pool4'))

        (self.feed('pool4')
             .zero_padding(paddings=1, name='padding13')
             .conv(3, 3, 512, 1, 1, name='conv5_1')
             .zero_padding(paddings=1, name='padding14')
             .conv(3, 3, 512, 1, 1, name='conv5_2')
             .zero_padding(paddings=1, name='padding15')
             .conv(3, 3, 512, 1, 1, name='conv5_3')
             .zero_padding(paddings=1, name='padding16')
             .max_pool(2, 2, 2, 2, name='pool5')
             .conv(7, 7, 4096, 1, 1, name='fc6')
             .conv(1, 1, 4096, 1, 1, name='fc7')
             .conv(1, 1, num_classes, 1, 1, name='score_fr')
             .deconv(4, 4, num_classes, 2, 2, name='upscore2'))

        (self.feed('upscore2', 'score_pool4')
             .crop(5, name='score_pool4c'))

        (self.feed('upscore2', 'score_pool4c')
             .add(name='fuse_pool4')
             .deconv(4, 4, num_classes, 2, 2, name='upscore_pool4'))

        (self.feed('upscore_pool4', 'score_pool3')
             .crop(9, name='score_pool3c'))

        (self.feed('upscore_pool4', 'score_pool3c')
             .add(name='fuse_pool3')
             .deconv(16, 16, num_classes, 8, 8, name='upscore8'))

        (self.feed('data', 'upscore8')
             .crop(31, name='score'))

        score = self.layers['score']
        score = tf.argmax(score, dimension=3)
        self.pred = decode_labels(score, self.input_size, num_classes)

    def read_input(self, img_path):
        self.img, self.filename = load_img(img_path)

    def forward(self, sess):
        return sess.run(self.pred, feed_dict={self.x: self.img})