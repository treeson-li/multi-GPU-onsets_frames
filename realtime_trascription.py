import collections
import os
import wave
import pyaudio
import struct
import numpy as np
import six
import scipy
import time
import queue
import librosa
import tensorflow as tf
import constants
import data
import model

from magenta.common import tf_utils
from magenta.music import audio_io
from magenta.music import midi_io
from magenta.music import sequences_lib
from magenta.protobuf import music_pb2

import real_time_constant as const
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'hparams',
    'batch_size=1',
    'A comma-separated list of `name=value` hyperparameter values.')
tf.app.flags.DEFINE_float(
    'frame_threshold', 0.5,
    'Threshold to use when sampling from the acoustic model.')
tf.app.flags.DEFINE_float(
    'onset_threshold', 0.5,
    'Threshold to use when sampling from the acoustic model.')

PLOT_LENGTH = 400

class Transcription:
    def __init__(self, checkpoint_dir):
        # create data 
        self.hparams = tf_utils.merge_hparams(constants.DEFAULT_HPARAMS, model.get_default_hparams())
        self.hparams.parse(FLAGS.hparams)
        self.checkpoint_dir = checkpoint_dir
        self.get_model()
        self.onset_frame = [np.zeros([PLOT_LENGTH, 88]), np.zeros([PLOT_LENGTH, 88]), np.zeros([PLOT_LENGTH, 88])]

    def parse_onset_frame(self):
        onset_frame = self.onset_frame
        # preprocess cut consistance frames according to onset
        temp = np.ones(onset_frame[const.ONSET].shape)
        for h in range(onset_frame[const.ONSET].shape[1]):
            for l in range(onset_frame[const.ONSET].shape[0]-1):
                if onset_frame[const.ONSET][l, h]<onset_frame[const.ONSET][l+1, h]: temp[l, h]=0
        onset_frame[const.FRAME] = temp * onset_frame[const.FRAME]
        sequence_prediction = sequences_lib.pianoroll_to_note_sequence(
            onset_frame[const.FRAME],
            frames_per_second=data.hparams_frames_per_second(self.hparams),
            min_duration_ms=0,
            onset_predictions=onset_frame[const.ONSET],
            velocity_values=onset_frame[const.VELOCITY])
        for note in sequence_prediction.notes: note.pitch=note.pitch+constants.MIN_MIDI_PITCH
        return sequences_lib.sequence_to_pianoroll(sequence_prediction, data.hparams_frames_per_second(self.hparams), 21, 108).active_velocities

    def create_example(self, samples):
        """Processes an audio file into an Example proto."""
        example_time = time.time()
        if self.hparams.normalize_audio:
            samples = librosa.util.normalize(samples)
        wav_data = audio_io.samples_to_wav_data(samples, self.hparams.sample_rate)

        example = tf.train.Example(features=tf.train.Features(feature={
            'id':
                tf.train.Feature(bytes_list=tf.train.BytesList(
                    value=['filename'.encode('utf-8')]
                )),
            'sequence':
                tf.train.Feature(bytes_list=tf.train.BytesList(
                    value=[music_pb2.NoteSequence().SerializeToString()]
                )),
            'audio':
                tf.train.Feature(bytes_list=tf.train.BytesList(
                    value=[wav_data]
                )),
            'velocity_range':
                tf.train.Feature(bytes_list=tf.train.BytesList(
                    value=[music_pb2.VelocityRange().SerializeToString()]
                )),
        }))
        print('|| example time:', time.time() - example_time)
        return example.SerializeToString()

    def restore_checkpoint(self, session, acoustic_checkpoint):
        var_all = tf.global_variables()
        var_to_restore = [var for var in var_all if not 'state_' in var.name]   # added variable to model
        var_to_init = [var for var in var_all if 'state_' in var.name] 
        print('------------------------ var to init', var_to_init)
        saver = tf.train.Saver(var_list=var_to_restore)
        saver.restore(session, acoustic_checkpoint)
        session.run(tf.variables_initializer(var_to_init))

    def get_model(self):
        """Initializes a transcription session."""
        with tf.Graph().as_default():
            self.examples = tf.placeholder(tf.string, [None])

            batch, self.iterator = data.provide_batch(
                batch_size=1,
                examples=self.examples,
                hparams=self.hparams,
                is_training=False,
                truncated_length=0)
            model.get_model(batch, self.hparams, is_training=False, is_real_time=True)

            self.session = tf.Session()
            self.restore_checkpoint(self.session, tf.train.latest_checkpoint(self.checkpoint_dir))
            
            self.onset_probs_flat = tf.get_default_graph().get_tensor_by_name(
                'onsets/onset_probs_flat:0')
            self.frame_probs_flat = tf.get_default_graph().get_tensor_by_name(
                'frame_probs_flat:0')
            self.velocity_values_flat = tf.get_default_graph().get_tensor_by_name(
                'velocity/velocity_values_flat:0')
            self.update_op = tf.get_collection('update_op')

    def transcrib(self, samples):
        print('|| transcribing ......')
        transcrib_time = time.time()
        self.session.run(self.iterator.initializer,
                        feed_dict={self.examples: [self.create_example(samples)]})
        
        frame_logits, onset_logits, velocity_values, _ = self.session.run([self.frame_probs_flat, self.onset_probs_flat, 
                                                                        self.velocity_values_flat, self.update_op])
        print('|| transcrib time:', time.time() - transcrib_time)
        frame_predictions = frame_logits > FLAGS.frame_threshold
        onset_predictions = onset_logits > FLAGS.onset_threshold
        piano_queue = [onset_predictions, frame_predictions, velocity_values]

        for i in range(3):
            self.onset_frame[i] = np.vstack((self.onset_frame[i], piano_queue[i]))
            self.onset_frame[i] = self.onset_frame[i][-PLOT_LENGTH:, :]
        return self.parse_onset_frame()

class Record_analog:
    def __init__(self):
        CHUNK = 512
        CHUNK_NUM = 5
        self.wav_dir = 'record/2018-11-6-test.wav'
        self.wav_data = librosa.core.load(self.wav_dir, sr=16000)[0]
        self.sample_size = CHUNK*(CHUNK_NUM+2*const.PADDING_ERROR)
        self.sample_step = CHUNK*CHUNK_NUM
        self.samples = np.zeros(self.sample_size)
        self.i = 0
        self.count = 0
    
    def recording(self):
        while True:
            if self.count >= len(self.wav_data)//(self.sample_step): self.count = 0
            time.sleep(0.1)
            self.i += 1
            self.count += 1

            self.samples = np.append(self.samples, self.wav_data[(self.count)*self.sample_step: (self.count+1)*self.sample_step])
            self.samples = self.samples[-self.sample_size:]
            yield (self.i, self.samples)
    
    def full_music(self):
        return self.wav_data

if __name__ == "__main__":
    checkpoint_dir = 'train2'
    sample_gen = Record_analog().recording()
    transcription = Transcription(checkpoint_dir)
    for i in range(10):
        sample = next(sample_gen)[1]
        re = transcription.transcrib(sample)
        # for i in re:
        print('====='*7)
        print(type(re), re.shape)
        print(re)
    