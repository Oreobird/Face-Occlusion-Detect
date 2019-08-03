import tensorflow as tf
from tensorflow.python.platform import app
import os
import numpy as np
import cv2
import argparse
import sys
import datasets
import models

def parse_args():
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    
    parser.add_argument("--proj_dir", type=str, default="./", help="Project directory")
    parser.add_argument("--input_size", type=int, default=96, help="Input image size")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size.")
    parser.add_argument("--fine_tune", type=bool, default=False, help="Fine tune based on Vgg16.")
    parser.add_argument("--train", type=bool, default=False, help="Train or test.")
    parser.add_argument("--epochs", type=int, default=100, help="Train epochs")
    parser.add_argument("--camera_test", type=bool, default=True, help="Camera video stream test. Need a camera device")
    
    return parser.parse_known_args()
    
def main(unused_args):
    if not len(FLAGS.proj_dir):
        raise Exception("Please set project directory")
    
    MODEL_DIR = os.path.join(FLAGS.proj_dir, 'model/')
    LOG_DIR = os.path.join(FLAGS.proj_dir, 'log/')
    
    FOD_CLASS_NAMES = ['normal', 'left_eye', 'right_eye', 'nose', 'mouth', 'chin']
    CLASS_NUM = len(FOD_CLASS_NAMES)
    

    dataset = datasets.Cofw(proj_dir=FLAGS.proj_dir, data_dir='data/cofw/', batch_size=FLAGS.batch_size,
                            input_size=FLAGS.input_size, class_num=CLASS_NUM,
                            fine_tune=FLAGS.fine_tune)
        
    net = models.FodNet(dataset, CLASS_NUM, batch_size=FLAGS.batch_size,
                        input_size=FLAGS.input_size, fine_tune=FLAGS.fine_tune,
                        fine_tune_model_file=os.path.join(MODEL_DIR, 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'))
    
    net.fit(MODEL_DIR + 'fod_model.h5', MODEL_DIR, LOG_DIR,
              max_epoches=FLAGS.epochs,
              train=FLAGS.train)
    
    if not FLAGS.camera_test:
        net.predict()
    else:
        import camera_tester
        tester = camera_tester.CameraTester(net, FLAGS.input_size, FLAGS.fine_tune,
                                            os.path.join(MODEL_DIR, 'shape_predictor_68_face_landmarks.dat'))
        tester.run()
        
if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)
    

