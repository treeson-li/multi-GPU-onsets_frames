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

"""Tests for shared data lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tempfile
import time

from magenta.models.onsets_frames_transcription import constants
from magenta.models.onsets_frames_transcription import data

from magenta.music import audio_io
from magenta.music import sequences_lib
from magenta.music import testing_lib
from magenta.protobuf import music_pb2

import numpy as np
import tensorflow as tf


class DataTest(tf.test.TestCase):

  def _FillExample(self, sequence, wav_data, filename):
    velocity_range = music_pb2.VelocityRange(min=0, max=127)
    feature_dict = {
        'id':
            tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[filename.encode('utf-8')])
            ),
        'sequence':
            tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[sequence.SerializeToString()])),
        'audio':
            tf.train.Feature(bytes_list=tf.train.BytesList(value=[wav_data])),
        'velocity_range':
            tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[velocity_range.SerializeToString()])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))

  def _DataToInputs(self, spec, labels, weighted_labels, length, filename,
                    truncated_length):
    del weighted_labels
    # This method re-implements a portion of the TensorFlow graph using numpy.
    # While typically it is frowned upon to test complicated code with other
    # code, there is no way around this for testing the pipeline end to end,
    # which requires an actual spec computation. Furthermore, much of the
    # complexity of the pipeline is due to the TensorFlow implementation,
    # so comparing it against simpler numpy code still provides effective
    # coverage.
    truncated_length = (
        min(truncated_length, length) if truncated_length else length)

    # Pad or slice spec if differs from truncated_length.
    if len(spec) < truncated_length:
      pad_amt = truncated_length - len(spec)
      spec = np.pad(spec, [(0, pad_amt), (0, 0)], 'constant')
    else:
      spec = spec[0:truncated_length]

    # Pad or slice labels if differs from truncated_length.
    if len(labels) < truncated_length:
      pad_amt = truncated_length - len(labels)
      labels = np.pad(labels, [(0, pad_amt), (0, 0)], 'constant')
    else:
      labels = labels[0:truncated_length]

    inputs = [(spec, labels, truncated_length, filename)]

    return inputs

  def _ExampleToInputs(self,
                       ex,
                       truncated_length=0,
                       crop_training_sequence_to_notes=False):
    hparams = copy.deepcopy(constants.DEFAULT_HPARAMS)
    hparams.crop_training_sequence_to_notes = crop_training_sequence_to_notes

    filename = ex.features.feature['id'].bytes_list.value[0]
    sequence, crop_beginning_seconds = data.preprocess_sequence(
        ex.features.feature['sequence'].bytes_list.value[0], hparams)
    wav_data = ex.features.feature['audio'].bytes_list.value[0]

    if crop_training_sequence_to_notes:
      wav_data = audio_io.crop_wav_data(wav_data, hparams.sample_rate,
                                        crop_beginning_seconds,
                                        sequence.total_time)
    spec = data.wav_to_spec(wav_data, hparams=hparams)
    roll = sequences_lib.sequence_to_pianoroll(
        sequence,
        frames_per_second=data.hparams_frames_per_second(hparams),
        min_pitch=constants.MIN_MIDI_PITCH,
        max_pitch=constants.MAX_MIDI_PITCH,
        min_frame_occupancy_for_label=0.0,
        onset_mode='length_ms',
        onset_length_ms=32.,
        onset_delay_ms=0.)
    length = data.wav_to_num_frames(
        wav_data, frames_per_second=data.hparams_frames_per_second(hparams))

    return self._DataToInputs(spec, roll.active, roll.weights, length, filename,
                              truncated_length)

  def _ValidateProvideBatch(self,
                            examples,
                            truncated_length,
                            batch_size,
                            expected_inputs,
                            crop_training_sequence_to_notes=False):
    """Tests for correctness of batches."""
    hparams = copy.deepcopy(constants.DEFAULT_HPARAMS)
    hparams.crop_training_sequence_to_notes = crop_training_sequence_to_notes

    with self.test_session() as sess:
      batch, _ = data.provide_batch(
          batch_size=batch_size,
          examples=examples,
          hparams=hparams,
          truncated_length=truncated_length,
          is_training=False)
      sess.run(tf.local_variables_initializer())
      input_tensors = [
          batch.spec, batch.labels, batch.lengths, batch.filenames,
          batch.max_length
      ]
      self.assertEqual(len(expected_inputs) // batch_size, batch.num_batches)
      for i in range(0, batch.num_batches * batch_size, batch_size):
        # Wait to ensure example is pre-processed.
        time.sleep(0.1)
        inputs = sess.run(input_tensors)
        max_length = np.max(inputs[2])
        self.assertEqual(inputs[4], max_length)
        inputs = inputs[0:-1]
        for j in range(batch_size):
          # Add batch padding if needed.
          input_length = expected_inputs[i + j][2]
          if input_length < max_length:
            expected_inputs[i + j] = list(expected_inputs[i + j])
            pad_amt = max_length - input_length
            expected_inputs[i + j][0] = np.pad(
                expected_inputs[i + j][0], [(0, pad_amt), (0, 0)], 'constant')
            expected_inputs[i + j][1] = np.pad(
                expected_inputs[i + j][1],
                [(0, pad_amt), (0, 0)], 'constant')
          for exp_input, input_ in zip(expected_inputs[i + j], inputs):
            self.assertAllEqual(np.squeeze(exp_input), np.squeeze(input_[j]))

      with self.assertRaisesOpError('End of sequence'):
        _ = sess.run(input_tensors)

  def _SyntheticSequence(self, duration, note, start_time=0):
    seq = music_pb2.NoteSequence(total_time=start_time + duration)
    testing_lib.add_track_to_sequence(
        seq, 0, [(note, 100, start_time, start_time + duration)])
    return seq

  def _ValidateProvideBatchTFRecord(self,
                                    truncated_length,
                                    batch_size,
                                    lengths,
                                    expected_num_inputs,
                                    crop_sequence_secs=0):
    hparams = copy.deepcopy(constants.DEFAULT_HPARAMS)
    examples = []
    expected_inputs = []

    for i, length in enumerate(lengths):
      wav_samples = np.zeros(
          (np.int((length / data.hparams_frames_per_second(hparams)) *
                  hparams.sample_rate), 1), np.float32)
      wav_data = audio_io.samples_to_wav_data(wav_samples, hparams.sample_rate)

      num_frames = data.wav_to_num_frames(
          wav_data, frames_per_second=data.hparams_frames_per_second(hparams))

      seq = self._SyntheticSequence(
          num_frames / data.hparams_frames_per_second(hparams) -
          crop_sequence_secs * 2,  # crop from both ends.
          i + constants.MIN_MIDI_PITCH,
          start_time=crop_sequence_secs)

      examples.append(self._FillExample(seq, wav_data, 'ex%d' % i))
      expected_inputs += self._ExampleToInputs(
          examples[-1],
          truncated_length,
          crop_training_sequence_to_notes=crop_sequence_secs > 0)
    self.assertEqual(expected_num_inputs, len(expected_inputs))

    with tempfile.NamedTemporaryFile() as temp_tfr:
      with tf.python_io.TFRecordWriter(temp_tfr.name) as writer:
        for ex in examples:
          writer.write(ex.SerializeToString())

      self._ValidateProvideBatch(
          temp_tfr.name,
          truncated_length,
          batch_size,
          expected_inputs,
          crop_training_sequence_to_notes=crop_sequence_secs > 0)

  def _ValidateProvideBatchMemory(self,
                                  truncated_length,
                                  batch_size,
                                  lengths,
                                  expected_num_inputs,
                                  crop_sequence_secs=0):
    hparams = copy.deepcopy(constants.DEFAULT_HPARAMS)
    examples = []
    expected_inputs = []

    for i, length in enumerate(lengths):
      wav_samples = np.zeros(
          (np.int((length / data.hparams_frames_per_second(hparams)) *
                  hparams.sample_rate), 1), np.float32)
      wav_data = audio_io.samples_to_wav_data(wav_samples, hparams.sample_rate)

      num_frames = data.wav_to_num_frames(
          wav_data, frames_per_second=data.hparams_frames_per_second(hparams))

      seq = self._SyntheticSequence(
          num_frames / data.hparams_frames_per_second(hparams) -
          crop_sequence_secs * 2,  # crop from both ends.
          i + constants.MIN_MIDI_PITCH,
          start_time=crop_sequence_secs)

      examples.append(self._FillExample(seq, wav_data, 'ex%d' % i))
      expected_inputs += self._ExampleToInputs(
          examples[-1],
          truncated_length,
          crop_training_sequence_to_notes=crop_sequence_secs > 0)
    self.assertEqual(expected_num_inputs, len(expected_inputs))

    self._ValidateProvideBatch(
        [e.SerializeToString() for e in examples],
        truncated_length,
        batch_size,
        expected_inputs,
        crop_training_sequence_to_notes=crop_sequence_secs > 0)

  def _ValidateProvideBatchBoth(self,
                                truncated_length,
                                batch_size,
                                lengths,
                                expected_num_inputs,
                                crop_sequence_secs=0):
    self._ValidateProvideBatchTFRecord(
        truncated_length=truncated_length,
        batch_size=batch_size,
        lengths=lengths,
        expected_num_inputs=expected_num_inputs,
        crop_sequence_secs=crop_sequence_secs)
    self._ValidateProvideBatchMemory(
        truncated_length=truncated_length,
        batch_size=batch_size,
        lengths=lengths,
        expected_num_inputs=expected_num_inputs,
        crop_sequence_secs=crop_sequence_secs)

  def testProvideBatchFullSeqs(self):
    self._ValidateProvideBatchBoth(
        truncated_length=0,
        batch_size=2,
        lengths=[10, 50, 100, 10, 50, 80],
        expected_num_inputs=6)

  def testProvideBatchTruncated(self):
    self._ValidateProvideBatchBoth(
        truncated_length=15,
        batch_size=2,
        lengths=[10, 50, 100, 10, 50, 80],
        expected_num_inputs=6)

  def testProvideBatchCropped(self):
    self._ValidateProvideBatchBoth(
        truncated_length=0,
        batch_size=2,
        lengths=[200, 200],
        expected_num_inputs=2,
        crop_sequence_secs=1)


if __name__ == '__main__':
  tf.test.main()