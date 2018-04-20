# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:22:16 2018

@author: acer
"""

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference_conv
import mnist_train_conv

BATCH_SIZE = 100

EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32,[BATCH_SIZE,mnist_inference_conv.IMAGE_SIZE,mnist_inference_conv.IMAGE_SIZE,mnist_inference_conv.NUM_CHANNELS],name='x-input')
        y_ = tf.placeholder(tf.float32,[None,mnist_inference_conv.OUTPUT_NODE],name='y-input')
        
        validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}
        
        y = mnist_inference_conv.inference(x,None,None)
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train_conv.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        
        
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train_conv.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy,feed_dict=validate_feed)
                    print("After %s training steps, validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)
            
            
def main(argv=None):
    mnist = input_data.read_data_sets("./",one_hot=True)
    evaluate(mnist)
    
if __name__ == '__main__':
    tf.app.run()
    
    
    