# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 17:31:36 2019

@author: shirhe-lyh


Convert tensorflow weights to pytorch weights for Xception models.

Reference:
    https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/
        tf_to_pytorch/convert_tf_to_pt/load_tf_weights.py
"""

import numpy as np
import tensorflow as tf
import torch


_BLOCK_UNIT_COUNT_MAP = {
    'xception_41': [[3, 1], [1, 8], [2, 1]],
    'xception_65': [[3, 1], [1, 16], [2, 1]],
    'xception_71': [[5, 1], [1, 16], [2, 1]],
}


def load_param(checkpoint_path, conversion_map, model_name):
    """Load parameters according to conversion_map.
    
    Args:
        checkpoint_path: Path to tensorflow's checkpoint file.
        conversion_map: A dictionary with format 
            {pytorch tensor in a model: checkpoint variable name}
        model_name: The name of Xception model, only supports 'xception_41',
            'xception_65', or 'xception_71'.
    """
    for pth_param, tf_param_name in conversion_map.items():
        tf_param_name = str(model_name) + '/' + tf_param_name
        tf_param = tf.train.load_variable(checkpoint_path, tf_param_name)
        if 'conv' in tf_param_name and 'weights' in tf_param_name:
            tf_param = np.transpose(tf_param, (3, 2, 0, 1))
            if 'depthwise' in tf_param_name:
                tf_param = np.transpose(tf_param, (1, 0, 2, 3))
        elif tf_param_name.endswith('weights'):
            tf_param = np.transpose(tf_param)
        assert pth_param.size() == tf_param.shape, ('Dimension mismatch: ' + 
            '{} vs {}; {}'.format(pth_param.size(), tf_param.shape, 
                 tf_param_name))
        pth_param.data = torch.from_numpy(tf_param)


def convert(model, checkpoint_path):
    """Load Pytorch Xception from TensorFlow checkpoint file.
    
    Args:
        model: The pytorch Xception model, only supports 'xception_41',
            'xception_65', or 'xception_71'.
        checkpoint_path: Path to tensorflow's checkpoint file.
        
    Returns:
        The pytorch Xception model with pretrained parameters.
    """
    block_unit_counts = _BLOCK_UNIT_COUNT_MAP.get(model.scope, None)
    if block_unit_counts is None:
        raise ValueError('Unsupported Xception model name.')
    flow_names = []
    block_indices = []
    unit_indices = []
    flow_names_unique = ['entry_flow', 'middle_flow', 'exit_flow']
    for i, [block_count, unit_count] in enumerate(block_unit_counts):
        flow_names += [flow_names_unique[i]] * (block_count * unit_count)
        for i in range(block_count):
            block_indices += [i + 1] * unit_count
            unit_indices += [j + 1 for j in range(unit_count)]
    
    conversion_map = {}
    # Root block
    conversion_map_for_root_block = {
        model._layers[0]._conv.weight: 
            'entry_flow/conv1_1/weights',
        model._layers[0]._batch_norm.bias: 
            'entry_flow/conv1_1/BatchNorm/beta',
        model._layers[0]._batch_norm.weight: 
            'entry_flow/conv1_1/BatchNorm/gamma',
        model._layers[0]._batch_norm.running_mean: 
            'entry_flow/conv1_1/BatchNorm/moving_mean',
        model._layers[0]._batch_norm.running_var: 
            'entry_flow/conv1_1/BatchNorm/moving_variance',
        model._layers[1]._conv.weight: 
            'entry_flow/conv1_2/weights',
        model._layers[1]._batch_norm.bias: 
            'entry_flow/conv1_2/BatchNorm/beta',
        model._layers[1]._batch_norm.weight: 
            'entry_flow/conv1_2/BatchNorm/gamma',
        model._layers[1]._batch_norm.running_mean: 
            'entry_flow/conv1_2/BatchNorm/moving_mean',
        model._layers[1]._batch_norm.running_var: 
            'entry_flow/conv1_2/BatchNorm/moving_variance',
    }
    conversion_map.update(conversion_map_for_root_block)
    
    # Xception block
    for i in range(len(model._layers[2]._blocks)):
        block = model._layers[2]._blocks[i]
        ind = [1, 3, 5]
        if len(block._separable_conv_block) < 6:
            ind = [0, 1, 2]
        for j in range(3):
            conversion_map_for_separable_block = {
                block._separable_conv_block[ind[j]]._conv_depthwise.weight:
                    ('{}/block{}/unit_{}/xception_module/' +
                     'separable_conv{}_depthwise/depthwise_weights').format(
                        flow_names[i], block_indices[i], unit_indices[i], j+1),
                block._separable_conv_block[ind[j]]._conv_pointwise.weight:
                    ('{}/block{}/unit_{}/xception_module/' +
                     'separable_conv{}_pointwise/weights').format(
                        flow_names[i], block_indices[i], unit_indices[i], j+1),
                block._separable_conv_block[ind[j]]._batch_norm_depthwise.bias:
                    ('{}/block{}/unit_{}/xception_module/' +
                     'separable_conv{}_depthwise/BatchNorm/beta').format(
                        flow_names[i], block_indices[i], unit_indices[i], j+1),
                block._separable_conv_block[ind[j]]._batch_norm_depthwise.weight:
                    ('{}/block{}/unit_{}/xception_module/' +
                     'separable_conv{}_depthwise/BatchNorm/gamma').format(
                        flow_names[i], block_indices[i], unit_indices[i], j+1),
                block._separable_conv_block[ind[j]]._batch_norm_depthwise.running_mean:
                    ('{}/block{}/unit_{}/xception_module/' +
                     'separable_conv{}_depthwise/BatchNorm/moving_mean').format(
                        flow_names[i], block_indices[i], unit_indices[i], j+1),
                block._separable_conv_block[ind[j]]._batch_norm_depthwise.running_var:
                    ('{}/block{}/unit_{}/xception_module/' +
                     'separable_conv{}_depthwise/BatchNorm/moving_variance').format(
                        flow_names[i], block_indices[i], unit_indices[i], j+1),
                block._separable_conv_block[ind[j]]._batch_norm_pointwise.bias:
                    ('{}/block{}/unit_{}/xception_module/' +
                     'separable_conv{}_pointwise/BatchNorm/beta').format(
                        flow_names[i], block_indices[i], unit_indices[i], j+1),
                block._separable_conv_block[ind[j]]._batch_norm_pointwise.weight:
                    ('{}/block{}/unit_{}/xception_module/' +
                     'separable_conv{}_pointwise/BatchNorm/gamma').format(
                        flow_names[i], block_indices[i], unit_indices[i], j+1),
                block._separable_conv_block[ind[j]]._batch_norm_pointwise.running_mean:
                    ('{}/block{}/unit_{}/xception_module/' +
                     'separable_conv{}_pointwise/BatchNorm/moving_mean').format(
                        flow_names[i], block_indices[i], unit_indices[i], j+1),
                block._separable_conv_block[ind[j]]._batch_norm_pointwise.running_var:
                    ('{}/block{}/unit_{}/xception_module/' +
                     'separable_conv{}_pointwise/BatchNorm/moving_variance').format(
                        flow_names[i], block_indices[i], unit_indices[i], j+1),
            }
            conversion_map.update(conversion_map_for_separable_block)
            
            if getattr(block, '_conv_skip_connection', None) is not None:
                conversion_map_for_shortcut = {
                    block._conv_skip_connection.weight:
                       ('{}/block{}/unit_{}/xception_module/shortcut/' +
                        'weights').format(
                            flow_names[i], block_indices[i], unit_indices[i]),
                    block._batch_norm_shortcut.bias:
                        ('{}/block{}/unit_{}/xception_module/shortcut/' +
                         'BatchNorm/beta').format(
                            flow_names[i], block_indices[i], unit_indices[i]),
                    block._batch_norm_shortcut.weight:
                        ('{}/block{}/unit_{}/xception_module/shortcut/' +
                         'BatchNorm/gamma').format(
                            flow_names[i], block_indices[i], unit_indices[i]),
                    block._batch_norm_shortcut.running_mean:
                        ('{}/block{}/unit_{}/xception_module/shortcut/' +
                         'BatchNorm/moving_mean').format(
                            flow_names[i], block_indices[i], unit_indices[i]),
                    block._batch_norm_shortcut.running_var:
                        ('{}/block{}/unit_{}/xception_module/shortcut/' +
                         'BatchNorm/moving_variance').format(
                            flow_names[i], block_indices[i], unit_indices[i]),
                }
                conversion_map.update(conversion_map_for_shortcut)
        
    # Prediction
    if len(model._layers) > 5:
        conversion_map_for_prediction = {
        model._layers[5].weight: 'logits/weights',
        model._layers[5].bias: 'logits/biases',
        }
        conversion_map.update(conversion_map_for_prediction)
        
    # Load TensorFlow parameters into PyTorch model
    load_param(checkpoint_path, conversion_map, model.scope)