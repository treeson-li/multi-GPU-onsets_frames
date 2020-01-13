# Copyright 2019 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Onset-focused model for piano transcription."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import constants
import real_time_constant as const

from magenta.common import flatten_maybe_padded_sequences
from magenta.common import tf_utils

import tensorflow as tf
import tensorflow.contrib.slim as slim

import thumt.layers.attention as transformer_att
import thumt.models.transformer as transformer

def get_state_variables(stack_multiplier, stack_size, batch_size, num_units):
    c = tf.zeros([stack_multiplier * stack_size, batch_size, num_units],
                 tf.float32)
    h = tf.zeros([stack_multiplier * stack_size, batch_size, num_units],
                 tf.float32)
    return (tf.Variable(h, trainable=False, name='state_h'), tf.Variable(c, trainable=False, name='state_c'))

def get_state_update_op(state_variables, new_states):
    # Add an operation to update the train states with the last state tensors
    update_ops = []
    update_ops.extend([state_variables[0][0, :, :].assign(new_states[0][0, :, :]),
                        state_variables[1][0, :, :].assign(new_states[1][0, :, :])])
    # Return a tuple in order to combine all update_ops into a single operation.
    # The tuple's actual value should not be used.
    return tf.tuple(update_ops, name='update_op')

def conv_net(inputs, hparams):
  """Builds the ConvNet from Kelz 2016."""
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      activation_fn=tf.nn.relu,
      weights_initializer=tf.contrib.layers.variance_scaling_initializer(
          factor=2.0, mode='FAN_AVG', uniform=True)):

    net = inputs
    i = 0
    for (conv_temporal_size, conv_freq_size,    #temporal_sizes:[3, 3, 3], freq_sizes:[3, 3, 3], num_filters:[48, 48, 96]
         num_filters, freq_pool_size, dropout_amt) in zip(  #[1, 2, 2], [1, 0.25. 0.25]
             hparams.temporal_sizes, hparams.freq_sizes, hparams.num_filters,
             hparams.pool_sizes, hparams.dropout_keep_amts):
      net = slim.conv2d(
          net,
          num_filters, [conv_temporal_size, conv_freq_size],
          scope='conv' + str(i),
          normalizer_fn=slim.batch_norm)
      if freq_pool_size > 1:
        net = slim.max_pool2d(
            net, [1, freq_pool_size],
            stride=[1, freq_pool_size],
            scope='pool' + str(i))
      if dropout_amt < 1:
        net = slim.dropout(net, dropout_amt, scope='dropout' + str(i))
      i += 1

    # Flatten while preserving batch and time dimensions.
    dims = tf.shape(net)
    net = tf.reshape(
        net, (dims[0], dims[1], net.shape[2].value * net.shape[3].value),
        'flatten_end')

    net = slim.fully_connected(net, hparams.fc_size, scope='fc_end')
    net = slim.dropout(net, hparams.fc_dropout_keep_amt, scope='dropout_end')

    return net


# def acoustic_model(inputs, hparams, lstm_units, lengths, is_real_time):
#   """Acoustic model that handles all specs for a sequence in one window."""
#   conv_output = conv_net_kelz(inputs)
#   if is_real_time:
#     conv_output = tf.slice(conv_output, [0, const.PADDING_ERROR, 0], [-1, tf.shape(conv_output)[1]-const.PADDING_ERROR*2, 768])

#   if lstm_units:
#     rnn_cell_fw = tf.contrib.rnn.LSTMBlockCell(lstm_units)
#     if is_real_time:
#         states = get_state_variables(hparams.batch_size, rnn_cell_fw)
#     else: states = None
#     if hparams.onset_bidirectional:
#       rnn_cell_bw = tf.contrib.rnn.LSTMBlockCell(lstm_units)
#       outputs, unused_output_states = tf.nn.bidirectional_dynamic_rnn(
#           rnn_cell_fw,
#           rnn_cell_bw,
#           inputs=conv_output,
#           sequence_length=lengths,
#           dtype=tf.float32)
#       combined_outputs = tf.concat(outputs, 2)
#     else:
#       combined_outputs, unused_output_states = tf.nn.dynamic_rnn(
#           rnn_cell_fw,
#           inputs=conv_output,
#           sequence_length=lengths,
#           dtype=tf.float32,
#           initial_state=states)
#     if is_real_time:
#         update_op = get_state_update_op(states, unused_output_states)
#         tf.add_to_collection('update_op', update_op)
#     return combined_outputs
def cudnn_lstm_layer(inputs,
                     batch_size,
                     num_units,
                     lengths=None,
                     stack_size=1,
                     rnn_dropout_drop_amt=0,
                     is_training=True,
                     bidirectional=True,
                     is_real_time=False):
  """Create a LSTM layer that uses cudnn."""
  inputs_t = tf.transpose(inputs, [1, 0, 2])    # [time , batch, unites]
  print('>>>>>>>>>>>>------------------------------ in, lengths is ', lengths)
  if lengths is not None:
    all_outputs = [inputs_t]
    for i in range(stack_size):
      with tf.variable_scope('stack_' + str(i)):
        with tf.variable_scope('forward'):
          lstm_fw = tf.contrib.cudnn_rnn.CudnnLSTM(
              num_layers=1,
              num_units=num_units,
              direction='unidirectional',
              dropout=rnn_dropout_drop_amt,
              kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
              ),
              bias_initializer=tf.zeros_initializer(),
          )

        c_fw = tf.zeros([1, batch_size, num_units], tf.float32)
        h_fw = tf.zeros([1, batch_size, num_units], tf.float32)

        outputs_fw, latest_state = lstm_fw(
            all_outputs[-1], (h_fw, c_fw), training=is_training)
        print('>>>>>>>>----------------------- outputs_fw', outputs_fw)
        print('>>>>>>>>----------------------- latest_state', latest_state)
        combined_outputs = outputs_fw

        if bidirectional:
          with tf.variable_scope('backward'):
            lstm_bw = tf.contrib.cudnn_rnn.CudnnLSTM(
                num_layers=1,
                num_units=num_units,
                direction='unidirectional',
                dropout=rnn_dropout_drop_amt,
                kernel_initializer=tf.contrib.layers
                .variance_scaling_initializer(),
                bias_initializer=tf.zeros_initializer(),
            )

          c_bw = tf.zeros([1, batch_size, num_units], tf.float32)
          h_bw = tf.zeros([1, batch_size, num_units], tf.float32)

          inputs_reversed = tf.reverse_sequence(
              all_outputs[-1], lengths, seq_axis=0, batch_axis=1)
          outputs_bw, _ = lstm_bw(
              inputs_reversed, (h_bw, c_bw), training=is_training)

          outputs_bw = tf.reverse_sequence(
              outputs_bw, lengths, seq_axis=0, batch_axis=1)

          combined_outputs = tf.concat([outputs_fw, outputs_bw], axis=2)

        all_outputs.append(combined_outputs)

    # for consistency with cudnn, here we just return the top of the stack,
    # although this can easily be altered to do other things, including be
    # more resnet like
    return tf.transpose(all_outputs[-1], [1, 0, 2])
  else:
    lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
        num_layers=stack_size,
        num_units=num_units,
        direction='bidirectional' if bidirectional else 'unidirectional',
        dropout=rnn_dropout_drop_amt,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
        bias_initializer=tf.zeros_initializer(),
    )
    stack_multiplier = 2 if bidirectional else 1
    if is_real_time:
        states = get_state_variables(stack_multiplier, stack_size, batch_size, num_units)
    else:
        c = tf.zeros([stack_multiplier * stack_size, batch_size, num_units],
                    tf.float32)
        h = tf.zeros([stack_multiplier * stack_size, batch_size, num_units],
                    tf.float32)
        states = (h, c)
    outputs, latest_state = lstm(inputs_t, (states), training=is_training)
    print('>>>>>>>>----------------------- outputs', outputs)
    print('>>>>>>>>----------------------- states', states)
    print('>>>>>>>>----------------------- latest_state', latest_state)
    print('>>>>>>>>----------------------- latest_state[1][0, :, :]', latest_state[1][0, :, :])
    if is_real_time:
        update_op = get_state_update_op(states, latest_state)
        tf.add_to_collection('update_op', update_op)
    outputs = tf.transpose(outputs, [1, 0, 2])

    return outputs


def lstm_layer(inputs,
               batch_size,
               num_units,
               lengths=None,
               stack_size=1,
               use_cudnn=False,
               rnn_dropout_drop_amt=0,
               is_training=True,
               bidirectional=True,
               is_real_time=False):
  """Create a LSTM layer using the specified backend."""
  if use_cudnn:
    return cudnn_lstm_layer(inputs, batch_size, num_units, lengths, stack_size,
                            rnn_dropout_drop_amt, is_training, bidirectional, is_real_time)
  else:
    assert rnn_dropout_drop_amt == 0
    cells_fw = [
        tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units)
        for _ in range(stack_size)
    ]
    cells_bw = [
        tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units)
        for _ in range(stack_size)
    ]
    with tf.variable_scope('cudnn_lstm'):
      (outputs, unused_state_f,
       unused_state_b) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
           cells_fw,
           cells_bw,
           inputs,
           dtype=tf.float32,
           sequence_length=lengths,
           parallel_iterations=1)

    return outputs

def self_attention_layer(inputs, num_units, params, is_training, bidirectional):
  """Create self sparse Attention Network layer."""
  return transformer.sparse_self_attention(inputs, num_units, params, is_training, bidirectional)

def acoustic_model(inputs, hparams, lstm_units, lengths, is_training=True, is_real_time=False):
  """Acoustic model that handles all specs for a sequence in one window."""
  with tf.variable_scope("acoustic_model"):
    conv_output = conv_net(inputs, hparams)
    if is_real_time:
      conv_output = conv_output[:, const.PADDING_ERROR:tf.shape(conv_output)[1]-const.PADDING_ERROR, :]
      tf.add_to_collection('conv_output', conv_output)
    print('>>>>>>>>----------------------- conv_output', conv_output)

    if hparams.use_transformer:
      return self_attention_layer(conv_output, lstm_units, hparams, is_training, hparams.bidirectional)

    if lstm_units:
      return lstm_layer(
          conv_output,
          hparams.batch_size,
          lstm_units,
          lengths=lengths if hparams.use_lengths else None,
          stack_size=hparams.acoustic_rnn_stack_size,
          use_cudnn=hparams.use_cudnn,
          is_training=is_training,
          bidirectional=hparams.bidirectional,
          is_real_time=is_real_time)

    else:
      return conv_output

def encoder_prepare(inputs, lstm_units, lengths, is_training, params, bidirectional=True, use_transformer=False):
  ''' prepare encoder for attention '''
  if use_transformer:
    enc_output = self_attention_layer(inputs, lstm_units, params, is_training, bidirectional)
  else:
    enc_output = lstm_layer(inputs, params.batch_size, lstm_units, 
                      lengths=lengths if params.use_lengths else None,
                      use_cudnn=params.use_cudnn, 
                      rnn_dropout_drop_amt=params.rnn_dropout_drop_amt, 
                      stack_size=params.encoder_rnn_stack_size,
                      is_training=is_training,
                      bidirectional=bidirectional)
    
  return enc_output

def get_model(transcription_data, hparams, is_training=True, is_real_time=False):
  """Builds the acoustic model."""
  if is_real_time:
    onset_labels = tf.zeros([tf.shape(transcription_data.spec)[0], tf.shape(transcription_data.spec)[1]-2*const.PADDING_ERROR, 88], 
                    tf.float32, 'onsets_ph')
    offset_labels = tf.zeros([tf.shape(transcription_data.spec)[0], tf.shape(transcription_data.spec)[1]-2*const.PADDING_ERROR, 88], 
                    tf.float32, 'onsets_ph')
    velocity_labels = tf.zeros([tf.shape(transcription_data.spec)[0], tf.shape(transcription_data.spec)[1]-2*const.PADDING_ERROR, 88],
                    tf.float32, 'velocities_ph')
    frame_labels = tf.zeros([tf.shape(transcription_data.spec)[0], tf.shape(transcription_data.spec)[1]-2*const.PADDING_ERROR, 88],
                    tf.float32, 'labels_ph')
    frame_label_weights = tf.zeros([tf.shape(transcription_data.spec)[0], tf.shape(transcription_data.spec)[1]-2*const.PADDING_ERROR, 88],
                    tf.float32, 'lable_weights_ph')
    lengths = tf.fill((1, ), tf.shape(transcription_data.spec)[1]-2*const.PADDING_ERROR, name='lengths_ph')
  else:
    onset_labels = transcription_data.onsets
    offset_labels = transcription_data.offsets
    velocity_labels = transcription_data.velocities
    frame_labels = transcription_data.labels
    frame_label_weights = transcription_data.label_weights
    lengths = transcription_data.lengths

  spec = transcription_data.spec

  if hparams.stop_activation_gradient and not hparams.activation_loss:
    raise ValueError(
        'If stop_activation_gradient is true, activation_loss must be true.')

  losses = {}
  if not is_training:
    hparams.att_dropout = 0
  #merged_labels = tf.concat([onset_labels, frame_label_weights, offset_labels], axis=2)
  with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
###########################################################################################################################################
###########################################################################################################################################
    with tf.variable_scope('onsets'):
      onset_label_enc = encoder_prepare(onset_labels,
                                  hparams.onset_lstm_units,
                                  lengths=lengths,
                                  is_training=is_training,
                                  params=hparams,
                                  bidirectional=True,
                                  use_transformer=False)
      
      spec_enc_4onset = acoustic_model(spec, hparams, hparams.spec_lstm_units, lengths, is_training)
      onset_outputs = transformer_att.sparse_multihead_attention(
                                  queries=spec_enc_4onset,
                                  memories=onset_label_enc,
                                  num_heads=hparams.num_heads,
                                  key_size=hparams.onset_lstm_units,
                                  value_size=hparams.onset_lstm_units,
                                  output_size=hparams.onset_lstm_units,
                                  keep_prob=1-hparams.att_dropout,
                                  scope='onset_attention',
                                  context=hparams.context,
                                  attn_mode='diag_band')
      onset_probs = slim.fully_connected(
          onset_outputs,
          constants.MIDI_PITCHES,
          activation_fn=tf.sigmoid,
          scope='onset_probs')
      # onset_probs_flat is used during inference.
      onset_probs_flat = flatten_maybe_padded_sequences(onset_probs, lengths)
      tf.add_to_collection('collection', onset_probs)
      tf.add_to_collection('collection', onset_probs_flat)
      onset_labels_flat = flatten_maybe_padded_sequences(onset_labels, lengths)
      tf.identity(onset_probs_flat, name='onset_probs_flat')
      tf.identity(onset_labels_flat, name='onset_labels_flat')
      tf.identity(
          tf.cast(tf.greater_equal(onset_probs_flat, .5), tf.float32),
          name='onset_predictions_flat')

      onset_losses = tf_utils.log_loss(onset_labels_flat, onset_probs_flat)
      tf.losses.add_loss(tf.reduce_mean(onset_losses))
      losses['onset'] = onset_losses
###########################################################################################################################################
###########################################################################################################################################
    with tf.variable_scope('offsets'):
      offset_lable_enc = encoder_prepare(offset_labels,
                                  hparams.offset_lstm_units,
                                  lengths=lengths,
                                  is_training=is_training,
                                  params=hparams,
                                  bidirectional=True,
                                  use_transformer=False)
      spec_enc_4offsets = acoustic_model(spec, hparams, lstm_units=hparams.offset_lstm_units, lengths=lengths, is_training=is_training)
      offset_outputs = transformer_att.sparse_multihead_attention(
                                  queries=spec_enc_4offsets,
                                  memories=offset_lable_enc,
                                  num_heads=hparams.num_heads,
                                  key_size=hparams.offset_lstm_units,
                                  value_size=hparams.offset_lstm_units,
                                  output_size=hparams.offset_lstm_units,
                                  keep_prob=1-hparams.att_dropout,
                                  scope='offset_attention',
                                  context=hparams.context,
                                  attn_mode='diag_band')
      offset_probs = slim.fully_connected(
          offset_outputs,
          constants.MIDI_PITCHES,
          activation_fn=tf.sigmoid,
          scope='offset_probs')
      # offset_probs_flat is used during inference.
      offset_probs_flat = flatten_maybe_padded_sequences(offset_probs, lengths)
      offset_labels_flat = flatten_maybe_padded_sequences(
          offset_labels, lengths)
      tf.identity(offset_probs_flat, name='offset_probs_flat')
      tf.identity(offset_labels_flat, name='offset_labels_flat')
      tf.identity(
          tf.cast(tf.greater_equal(offset_probs_flat, .5), tf.float32),
          name='offset_predictions_flat')

      offset_losses = tf_utils.log_loss(offset_labels_flat, offset_probs_flat)
      tf.losses.add_loss(tf.reduce_mean(offset_losses))
      losses['offset'] = offset_losses
###########################################################################################################################################
###########################################################################################################################################
    with tf.variable_scope('velocity'):
      velocity_outputs = acoustic_model(
          spec,
          hparams,
          lstm_units=hparams.velocity_lstm_units,
          lengths=lengths,
          is_training=is_training,
          is_real_time=is_real_time)
      velocity_values = slim.fully_connected(
          velocity_outputs,
          constants.MIDI_PITCHES,
          activation_fn=None,
          scope='onset_velocities')
      
      velocity_values_flat = flatten_maybe_padded_sequences(
          velocity_values, lengths)
      tf.identity(velocity_values_flat, name='velocity_values_flat')
      velocity_labels_flat = flatten_maybe_padded_sequences(
          velocity_labels, lengths)
      velocity_loss = tf.reduce_sum(
          onset_labels_flat *
          tf.square(velocity_labels_flat - velocity_values_flat),
          axis=1)
      tf.losses.add_loss(tf.reduce_mean(velocity_loss))
      losses['velocity'] = velocity_loss
###########################################################################################################################################
###########################################################################################################################################
    with tf.variable_scope('frame'):
      if not hparams.share_conv_features:
        # TODO(eriche): this is broken when hparams.frame_lstm_units > 0
        activation_outputs = acoustic_model(
            spec,
            hparams,
            lstm_units=hparams.frame_lstm_units,
            lengths=lengths,
            is_training=is_training,
            is_real_time=is_real_time)
        activation_probs = slim.fully_connected(
            activation_outputs,
            constants.MIDI_PITCHES,
            activation_fn=tf.sigmoid,
            scope='activation_probs')
      else:
        activation_probs = slim.fully_connected(
            onset_outputs,
            constants.MIDI_PITCHES,
            activation_fn=tf.sigmoid,
            scope='activation_probs')

      probs = []
      if hparams.stop_onset_gradient:
        probs.append(tf.stop_gradient(onset_probs))
      else:
        probs.append(onset_probs)

      if hparams.stop_activation_gradient:
        probs.append(tf.stop_gradient(activation_probs))
      else:
        probs.append(activation_probs)

      if hparams.stop_offset_gradient:
        probs.append(tf.stop_gradient(offset_probs))
      else:
        probs.append(offset_probs)

      combined_probs = tf.concat(probs, 2)

      if hparams.combined_lstm_units > 0:
        frame_label_enc = encoder_prepare(frame_label_weights,
                                  hparams.combined_lstm_units,                                                
                                  lengths,
                                  is_training=is_training,
                                  params=hparams,
                                  bidirectional=True,
                                  use_transformer=False)
        frame_query_enc = encoder_prepare(
                                  combined_probs,
                                  hparams.combined_lstm_units,
                                  lengths=lengths,
                                  is_training=is_training,
                                  params=hparams,
                                  bidirectional=hparams.bidirectional,
                                  use_transformer=False)
        frame_outputs = transformer_att.sparse_multihead_attention(
                                  queries=frame_query_enc,
                                  memories=frame_label_enc,
                                  num_heads=hparams.num_heads,
                                  key_size=hparams.combined_lstm_units,
                                  value_size=hparams.combined_lstm_units,
                                  output_size=hparams.combined_lstm_units,
                                  keep_prob=1-hparams.att_dropout,
                                  scope='frame_attention',
                                  context=hparams.context,
                                  attn_mode='diag_band')  
      else:
        frame_outputs = combined_probs

      frame_probs = slim.fully_connected(
          frame_outputs,
          constants.MIDI_PITCHES,
          activation_fn=tf.sigmoid,
          scope='frame_probs')
      
    frame_labels_flat = flatten_maybe_padded_sequences(frame_labels, lengths)
    frame_probs_flat = flatten_maybe_padded_sequences(frame_probs, lengths)
    tf.identity(frame_probs_flat, name='frame_probs_flat')
    frame_label_weights_flat = flatten_maybe_padded_sequences(
        frame_label_weights, lengths)
    if hparams.weight_frame_and_activation_loss:
      frame_loss_weights = frame_label_weights_flat
    else:
      frame_loss_weights = None
    frame_losses = tf_utils.log_loss(
        frame_labels_flat,
        frame_probs_flat,
        weights=frame_loss_weights)
    tf.losses.add_loss(tf.reduce_mean(frame_losses))
    losses['frame'] = frame_losses

    if hparams.activation_loss:
      if hparams.weight_frame_and_activation_loss:
        activation_loss_weights = frame_label_weights
      else:
        activation_loss_weights = None
      activation_losses = tf_utils.log_loss(
          frame_labels_flat,
          flatten_maybe_padded_sequences(activation_probs, lengths),
          weights=activation_loss_weights)
      tf.losses.add_loss(tf.reduce_mean(activation_losses))
      losses['activation'] = activation_losses

###########################################################################################################################################
###########################################################################################################################################
    with tf.variable_scope('spec'):
      fussion = tf.concat([onset_probs, offset_probs, frame_probs], axis=2)
      
      fuss_output = encoder_prepare(
                                  fussion,
                                  hparams.combined_lstm_units,
                                  lengths=lengths,
                                  is_training=is_training,
                                  params=hparams,
                                  bidirectional=hparams.bidirectional,
                                  use_transformer=False)      
      spec_bins = 229
      spec_dynamic = slim.fully_connected(
          fuss_output,
          constants.MIDI_PITCHES * hparams.template_num,
          activation_fn=tf.sigmoid,
          scope='spec_dynamic')
      
      dims = tf.shape(spec_dynamic)
      init_uniform = tf.random_uniform_initializer(minval=0, maxval=0.1, seed=None, dtype=tf.float32)
      dynamic_weight = tf.reshape(spec_dynamic, (dims[0], dims[1], constants.MIDI_PITCHES, hparams.template_num, 1), 'dynamic_weight')
      key_template = tf.get_variable('template', shape=[constants.MIDI_PITCHES, hparams.template_num, spec_bins], dtype=tf.float32, initializer=init_uniform)
      spec_output = tf.multiply(dynamic_weight, key_template)
      spec_output = tf.reduce_sum(spec_output, axis=3)
      spec_output = tf.reduce_sum(spec_output, axis=2)
      spec_output = tf.sigmoid(spec_output, name="spec_output")
      
      spec_out_flat = flatten_maybe_padded_sequences(spec_output, lengths)
      # spec_out_flat is not used during inference.
      if is_training:        
        spec_labels_flat = flatten_maybe_padded_sequences(spec, lengths)
        spec_labels_flat = tf.reshape(spec_out_flat, (-1, spec_bins))
        spec_losses = tf_utils.log_loss(spec_labels_flat, spec_out_flat)
        tf.losses.add_loss(tf.reduce_mean(spec_losses))
        losses['spec'] = spec_losses


  predictions_flat = tf.cast(tf.greater_equal(frame_probs_flat, .5), tf.float32)

  # Creates a pianoroll labels in red and probs in green [minibatch, 88]
  images = {}
  onset_pianorolls = tf.concat(
      [
          onset_labels[:, :, :, tf.newaxis], onset_probs[:, :, :, tf.newaxis],
          tf.zeros(tf.shape(onset_labels))[:, :, :, tf.newaxis]
      ],
      axis=3)
  images['OnsetPianorolls'] = onset_pianorolls
  offset_pianorolls = tf.concat([
      offset_labels[:, :, :, tf.newaxis], offset_probs[:, :, :, tf.newaxis],
      tf.zeros(tf.shape(offset_labels))[:, :, :, tf.newaxis]
  ],
                                axis=3)
  images['OffsetPianorolls'] = offset_pianorolls
  activation_pianorolls = tf.concat(
      [
          frame_labels[:, :, :, tf.newaxis], frame_probs[:, :, :, tf.newaxis],
          tf.zeros(tf.shape(frame_labels))[:, :, :, tf.newaxis]
      ],
      axis=3)
  images['ActivationPianorolls'] = activation_pianorolls

  return (tf.losses.get_total_loss(), losses, frame_labels_flat,
          predictions_flat, images)


def get_default_hparams():
  """Returns the default hyperparameters.

  Returns:
    A tf.contrib.training.HParams object representing the default
    hyperparameters for the model.
  """
  return tf.contrib.training.HParams(
      batch_size=5,
      spec_fmin=30.0,
      spec_n_bins=229,
      spec_type='mel',
      spec_mel_htk=True,
      spec_log_amplitude=True,
      transform_audio=True,
      learning_rate=0.0006,
      clip_norm=3,
      truncated_length=0,#1500,  # 48 seconds
      spec_lstm_units=256,
      onset_lstm_units=256,
      offset_lstm_units=256,
      velocity_lstm_units=0,
      frame_lstm_units=0,
      combined_lstm_units=256,
      onset_mode='length_ms',
      acoustic_rnn_stack_size=1,
      combined_rnn_stack_size=1,
      encoder_rnn_stack_size=1,
      # using this will result in output not aligning with audio.
      backward_shift_amount_ms=0,
      activation_loss=False,
      stop_activation_gradient=False,
      onset_length=32,
      offset_length=32,
      decay_steps=10000,
      decay_rate=0.98,
      stop_onset_gradient=True,
      stop_offset_gradient=True,
      weight_frame_and_activation_loss=False,
      share_conv_features=False,
      temporal_sizes=[3, 3, 3],
      freq_sizes=[3, 3, 3],
      num_filters=[48, 48, 96],
      pool_sizes=[1, 2, 2],
      dropout_keep_amts=[1.0, 0.25, 0.25],
      fc_size=768,
      fc_dropout_keep_amt=0.5,
      use_lengths=False,
      use_cudnn=True,
      rnn_dropout_drop_amt=0.0,
      bidirectional=False,#True,
      onset_overlap=True,
      preprocess_examples=False,

      use_transformer=False,#True,#
      num_heads=4,#1,#8,#
      att_dropout=0.1,
      use_ffn=False,
      layer_preprocess="layer_norm",
      layer_postprocess="layer_norm",
      context=128,
      template_num = 4
  )
