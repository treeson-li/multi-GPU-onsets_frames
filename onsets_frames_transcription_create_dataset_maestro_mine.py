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
"""Create the recordio files necessary for training onsets and frames.

The training files are split in ~20 second chunks by default, the test files
are not split.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import re

import librosa
from magenta.models.onsets_frames_transcription import create_dataset_util
from magenta.music import audio_io
from magenta.music import midi_io
from magenta.music import sequences_lib
from magenta.protobuf import music_pb2
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_dir', '/media/admin1/32B44FF2B44FB75F/Data/maestro/maestro-v1.0.0',
                           'Directory where the un-zipped MAPS files are.')
tf.app.flags.DEFINE_string('output_dir', './data/',
                           'Directory where the two output TFRecord files '
                           '(train and test) will be placed.')
tf.app.flags.DEFINE_integer('min_length', 5, 'minimum segment length')
tf.app.flags.DEFINE_integer('max_length', 20, 'maximum segment length')
tf.app.flags.DEFINE_integer('sample_rate', 16000, 'desired sample rate')

TEST_DIRS = ['ENSTDkCl/MUS', 'ENSTDkAm/MUS']
TRAIN_DIRS = ['AkPnBcht/MUS', 'AkPnBsdf/MUS', 'AkPnCGdD/MUS', 'AkPnStgb/MUS',
              'SptkBGAm/MUS', 'SptkBGCl/MUS', 'StbgTGd2/MUS']


def filename_to_id(filename):
  """Translate a .wav or .mid path to a MAPS sequence id."""
  return re.match(r'.*MUS-(.*)_[^_]+\.\w{3}',
                  os.path.basename(filename)).group(1)

def get_pair_data(pair_type):
  import csv
  maestro_dir = FLAGS.input_dir
  config_dir = os.path.join(maestro_dir, 'maestro-v1.0.0.csv')
  pair = []
  
  with open(config_dir) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
      if row[2] == pair_type:
        wav_dir = os.path.join(maestro_dir, row[5])
        midi_dir = os.path.join(maestro_dir, row[4])
        pair.append((wav_dir, midi_dir))
  return pair


def generate_train_set():
  """Generate the train TFRecord."""
  train_file_pairs = get_pair_data('train')
  print('************************************ train_file_pairs', len(train_file_pairs), '\n')
  train_output_name = os.path.join(FLAGS.output_dir,
                                   'maestro_train.tfrecord')

  with tf.python_io.TFRecordWriter(train_output_name) as writer:
    for idx, pair in enumerate(train_file_pairs):
      print('{} of {}: {}'.format(idx, len(train_file_pairs), pair[0]))
      # load the wav data
      wav_data = tf.gfile.Open(pair[0], mode='rb').read()
      samples = audio_io.wav_data_to_samples(wav_data, FLAGS.sample_rate)
      norm_samples = librosa.util.normalize(samples, norm=np.inf)

      # load the midi data and convert to a notesequence
      ns = midi_io.midi_file_to_note_sequence(pair[1])

      splits = create_dataset_util.find_split_points(
          ns, norm_samples, FLAGS.sample_rate, FLAGS.min_length,
          FLAGS.max_length)

      velocities = [note.velocity for note in ns.notes]
      velocity_max = np.max(velocities)
      velocity_min = np.min(velocities)
      new_velocity_tuple = music_pb2.VelocityRange(
          min=velocity_min, max=velocity_max)

      for start, end in zip(splits[:-1], splits[1:]):
        # print(start, end)
        if end - start < FLAGS.min_length:
          continue

        new_ns = sequences_lib.extract_subsequence(ns, start, end)
        samples_start = int(start * FLAGS.sample_rate)
        samples_end = samples_start + int((end-start) * FLAGS.sample_rate)
        new_samples = samples[samples_start:samples_end]
        new_wav_data = audio_io.samples_to_wav_data(new_samples,
                                                    FLAGS.sample_rate)

        example = tf.train.Example(features=tf.train.Features(feature={
            'id':
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[pair[0].encode()]
                )),
            'sequence':
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[new_ns.SerializeToString()]
                )),
            'audio':
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[new_wav_data]
                )),
            'velocity_range':
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[new_velocity_tuple.SerializeToString()]
                )),
            }))
        writer.write(example.SerializeToString())
    #   break


def generate_test_set():
  """Generate the test TFRecord."""
  test_file_pairs = get_pair_data('test')
  print('************************************ test_file_pairs', len(test_file_pairs), '\n')
  test_output_name = os.path.join(FLAGS.output_dir,
                                  'maestro_test.tfrecord')
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

      velocities = [note.velocity for note in ns.notes]
      velocity_max = np.max(velocities)
      velocity_min = np.min(velocities)
      new_velocity_tuple = music_pb2.VelocityRange(
          min=velocity_min, max=velocity_max)

      example = tf.train.Example(features=tf.train.Features(feature={
          'id':
          tf.train.Feature(bytes_list=tf.train.BytesList(
              value=[pair[0].encode()]
              )),
          'sequence':
          tf.train.Feature(bytes_list=tf.train.BytesList(
              value=[ns.SerializeToString()]  # method of protocol buffers
              )),
          'audio':
          tf.train.Feature(bytes_list=tf.train.BytesList(
              value=[wav_data]
              )),
          'velocity_range':
          tf.train.Feature(bytes_list=tf.train.BytesList(
              value=[new_velocity_tuple.SerializeToString()]  # # method of protocol buffers
              )),
          }))
      writer.write(example.SerializeToString())
    #   break

def generate_val_set():
  """Generate the val TFRecord."""
  val_file_pairs = get_pair_data('validation')
  print('************************************ val_file_pairs', len(val_file_pairs), '\n')
  val_output_name = os.path.join(FLAGS.output_dir,
                                  'maestro_val.tfrecord')
  with tf.python_io.TFRecordWriter(val_output_name) as writer:
    for idx, pair in enumerate(val_file_pairs):
      print('{} of {}: {}'.format(idx, len(val_file_pairs), pair[0]))
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

      velocities = [note.velocity for note in ns.notes]
      velocity_max = np.max(velocities)
      velocity_min = np.min(velocities)
      new_velocity_tuple = music_pb2.VelocityRange(
          min=velocity_min, max=velocity_max)

      example = tf.train.Example(features=tf.train.Features(feature={
          'id':
          tf.train.Feature(bytes_list=tf.train.BytesList(
              value=[pair[0].encode()]
              )),
          'sequence':
          tf.train.Feature(bytes_list=tf.train.BytesList(
              value=[ns.SerializeToString()]  # method of protocol buffers
              )),
          'audio':
          tf.train.Feature(bytes_list=tf.train.BytesList(
              value=[wav_data]
              )),
          'velocity_range':
          tf.train.Feature(bytes_list=tf.train.BytesList(
              value=[new_velocity_tuple.SerializeToString()]  # # method of protocol buffers
              )),
          }))
      writer.write(example.SerializeToString())
    #   break

def main(unused_argv):
  generate_train_set()
#   generate_test_set()
  generate_val_set()


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
