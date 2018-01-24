from __future__ import print_function

import argparse
import os
import sys
import time
from PIL import Image
import tensorflow as tf
import numpy as np
from scipy import misc

from model import FCN8s
from tools import *

save_dir = './output/'
model_path = './model/fcn_iter_160000.npy'

def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced PSPNet")
    parser.add_argument("--img-path", type=str, default='',
                        help="Path to the RGB image file.",
                        required=True)
    parser.add_argument("--save-dir", type=str, default=save_dir,
                        help="Path to save output.")

    return parser.parse_args()

def main():
    args = get_arguments()

    model = FCN8s()
    model.read_input(args.img_path)

    # Init tf Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    model.load(model_path, sess)

    preds = model.forward(sess)
      
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    misc.imsave(args.save_dir + model.filename, preds[0])
    
if __name__ == '__main__':
    main()
