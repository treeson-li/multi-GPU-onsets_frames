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

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create the tfrecord files necessary for training onsets and frames.

The training files are split in ~20 second chunks by default, the test files
are not split.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import re

import split_audio_and_label_data

from magenta.music import audio_io
from magenta.music import midi_io

import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_dir', '/media/admin1/32B44FF2B44FB75F/Data/MAPS/*/',
                           'Directory where the un-zipped MAPS files are.')
tf.app.flags.DEFINE_string('output_dir', './data/',
                           'Directory where the two output TFRecord files '
                           '(train and test) will be placed.')
tf.app.flags.DEFINE_integer('min_length', 5, 'minimum segment length')
tf.app.flags.DEFINE_integer('max_length', 20, 'maximum segment length')
tf.app.flags.DEFINE_integer('sample_rate', 16000, 'desired sample rate')

test_dirs = ['ENSTDkCl/MUS', 'ENSTDkAm/MUS']
train_dirs = [
    'AkPnBcht/MUS', 'AkPnBsdf/MUS', 'AkPnCGdD/MUS', 'AkPnStgb/MUS',
    'SptkBGAm/MUS', 'SptkBGCl/MUS', 'StbgTGd2/MUS'
]


def filename_to_id(filename):
  """Translate a .wav or .mid path to a MAPS sequence id."""
  return re.match(r'.*MUS-(.*)_[^_]+\.\w{3}',
                  os.path.basename(filename)).group(1)


def generate_train_set(exclude_ids):
  """Generate the train TFRecord."""
  train_file_pairs = []
  for directory in train_dirs:
    path = os.path.join(FLAGS.input_dir, directory)
    path = os.path.join(path, '*.wav')
    wav_files = glob.glob(path)
    # find matching mid files
    for wav_file in wav_files:
      base_name_root, _ = os.path.splitext(wav_file)
      mid_file = base_name_root + '.mid'
      if filename_to_id(wav_file) not in exclude_ids:
        train_file_pairs.append((wav_file, mid_file))
  print('************************************ train_file_pairs', len(train_file_pairs), '\n')
  train_output_name = os.path.join(FLAGS.output_dir,
                                   'maps_config2_train.tfrecord')

  with tf.python_io.TFRecordWriter(train_output_name) as writer:
    for idx, pair in enumerate(train_file_pairs):
      print('{} of {}: {}'.format(idx, len(train_file_pairs), pair[0]))
      # load the wav data
      wav_data = tf.gfile.Open(pair[0], 'rb').read()
      # load the midi data and convert to a notesequence
      ns = midi_io.midi_file_to_note_sequence(pair[1])
      for example in split_audio_and_label_data.process_record(
          wav_data, ns, pair[0], FLAGS.min_length, FLAGS.max_length,
          FLAGS.sample_rate):
        writer.write(example.SerializeToString())
      break


def generate_test_set():
  """Generate the test TFRecord."""
  test_file_pairs = []
  for directory in test_dirs:
    path = os.path.join(FLAGS.input_dir, directory)
    path = os.path.join(path, '*.wav')
    wav_files = glob.glob(path)
    # find matching mid files
    for wav_file in wav_files:
      base_name_root, _ = os.path.splitext(wav_file)
      mid_file = base_name_root + '.mid'
      test_file_pairs.append((wav_file, mid_file))
  print('================================>>> test file pairs info ', len(test_file_pairs), test_file_pairs[0][0])
  test_output_name = os.path.join(FLAGS.output_dir,
                                  'maps_config2_test.tfrecord')
  print('================================>>> test_output_name', test_output_name)
  with tf.python_io.TFRecordWriter(test_output_name) as writer:
    for idx, pair in enumerate(test_file_pairs):
      print('{} of {}: {}'.format(idx, len(test_file_pairs), pair[0]))
      # load the wav data and resample it.
      samples = audio_io.load_audio(pair[0], FLAGS.sample_rate)
      # print('???????????samples', type(samples), samples.shape, (samples[:44] * np.iinfo(np.int16).max).astype(np.int16))
      # import librosa
      # dataa, _ = librosa.load(pair[0], FLAGS.sample_rate)
      # print('???????????dataa', type(dataa), dataa.shape, dataa[:3])   # upper 2line code is the same as this annotation
      wav_data = audio_io.samples_to_wav_data(samples, FLAGS.sample_rate) # bytes type of wav
      # print('???????????wav_data', type(wav_data), len(wav_data), wav_data[:3])

      # load the midi data and convert to a notesequence
      ns = midi_io.midi_file_to_note_sequence(pair[1])

      example = split_audio_and_label_data.create_example(pair[0], ns, wav_data)
      writer.write(example.SerializeToString())
      break

  return [filename_to_id(wav) for wav, _ in test_file_pairs]


def main(unused_argv):
  print('##############################################################')
  test_ids = generate_test_set()
  # test_ids = []
  generate_train_set(test_ids)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
