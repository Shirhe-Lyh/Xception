# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 20:57:33 2019

@author: shirhe-lyh
"""

import argparse
import cv2
import numpy as np
import os
import torch
import tensorflow as tf

import converter
import original_tf.xception as xception_tf
import xception

slim = tf.contrib.slim

parser = argparse.ArgumentParser(
    description='Test pytorch results whether or not equal to tensorflow`s')
parser.add_argument('--tf_checkpoint_path', type=str, default=None,
                    help='Path to checkpoint file.')
parser.add_argument('--output_dir', type=str, default='./pretrained_models',
                    help='Where the output pytorch model file is stored.')
parser.add_argument('--image_path', type=str, default='./test/cat.jpg',
                    help='Path to a test image.')
parser.add_argument('--model_name', type=str, default='xception_65',
                    help='One of [xception_41, xception_65, xception_71].')
args = parser.parse_args()


def resize_and_crop(image, size=299):
    """Resize image and center crop."""
    if image is None:
        raise ValueError('image must be specified.')
        
    height, width, _ = image.shape
    if height > width:
        height = int(height * size / width)
        width = size
    else:
        width = int(width * size / height)
        height = size
    image = cv2.resize(image, (width, height))
    h = (height - size) // 2
    w = (width - size) // 2
    return image[h:h+size, w:w+size]


if __name__ == '__main__':
    image_path = args.image_path
    checkpoint_path = args.tf_checkpoint_path
    output_dir = args.output_dir
    model_name = args.model_name
    
    if not os.path.exists(image_path):
        raise ValueError('`image_path` does not exist.')
    if not os.path.exists(checkpoint_path + '.index'):
        raise ValueError('`tf_checkpoint_path` does not exit.')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'xception_65.pth')
    
    image = cv2.imread(image_path)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = resize_and_crop(image_rgb, size=299)
    image_center = (2.0 / 255) * image_rgb - 1.0
    image_center = image_center.astype(np.float32)
    images = np.expand_dims(image_center, axis=0)
    images_pth = np.expand_dims(np.transpose(image_center, axes=(2, 0, 1)),
                               axis=0)
    images_pth = torch.from_numpy(images_pth)
    
    
    # TensorFlow predicion
    inputs = tf.placeholder(tf.float32, shape=[None, 299, 299, 3], name='inputs')
    
    with slim.arg_scope(xception_tf.xception_arg_scope()):
        if model_name == 'xception_41':
            net, end_points = xception_tf.xception_41(inputs, num_classes=1001,
                                                      is_training=False)
        elif model_name == 'xception_71':
            net, end_points = xception_tf.xception_71(inputs, num_classes=1001,
                                                      is_training=False)
        else:
            net, end_points = xception_tf.xception_65(inputs, num_classes=1001,
                                                      is_training=False)
    predictions = tf.squeeze(end_points.get('predictions'), axis=[1, 2])
    classes = tf.argmax(predictions, axis=1)
    
    init = tf.global_variables_initializer()
    
    saver = tf.train.Saver(var_list=slim.get_model_variables())
    
    with tf.Session() as sess:
        sess.run(init)
        
        # Load tensorflow pretrained paremeters
        saver.restore(sess, checkpoint_path)
        
        logits, labels = sess.run([predictions, classes], 
                                  feed_dict={inputs: images})
        print('TensorFlow predicion:')
        print(labels)
        print(np.argsort(logits)[:, -5:])
    
    
    # PyTorch prediction
    if model_name == 'xception_41':
        model = xception.Xception41(num_classes=1001)
    elif model_name == 'xception_71':
        model = xception.Xception71(num_classes=1001)
    else:
        model = xception.Xception65(num_classes=1001)
    converter.convert(model, checkpoint_path)
    model.eval()
    with torch.no_grad():
        logits_pth = torch.nn.functional.softmax(model(images_pth), dim=1)
        logits_pth = logits_pth.data.cpu().numpy().squeeze(axis=2).squeeze(axis=2)
        labels_pth = np.argmax(logits_pth, axis=1)
        print('PyTorch prediction:')
        print(labels_pth)
        print(np.argsort(logits_pth)[:, -5:])
        
    
    # Save pytorch file
    torch.save(model.state_dict(), output_path)
    print('Save model to: ', output_path)
    
    
    # Test converted xception model
    if model_name == 'xception_41':
        xception_ = xception.xception_41(num_classes=1001, 
                                         pretrained=True,
                                         checkpoint_path=output_path)
    elif model_name == 'xception_71':
        xception_ = xception.xception_71(num_classes=1001, 
                                         pretrained=True,
                                         checkpoint_path=output_path)
    else:
        xception_ = xception.xception_65(num_classes=1001, 
                                         pretrained=True,
                                         checkpoint_path=output_path)

    xception_.eval()
    with torch.no_grad():
        logits_pth = torch.nn.functional.softmax(xception_(images_pth), dim=1)
        logits_pth = logits_pth.data.cpu().numpy().squeeze(axis=2).squeeze(axis=2)
        labels_pth = np.argmax(logits_pth, axis=1)
        print('PyTorch prediction:')
        print(labels_pth)
        print(np.argsort(logits_pth)[:, -5:])
