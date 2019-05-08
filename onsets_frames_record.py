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
# 定义数据流块
# CHUNK = 1024

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 512  # 0.03 seconds 
# 录音时间
CHUNK_NUM = 5  # 0.128 seconds
ABANDON_SPEC = 6 # first several spec result to abandon
# 要写入的文件名
WAVE_OUTPUT_FILENAME = "output.wav"
MIDI_OUTPUT_FILENAME = "output.mid"
# 创建PyAudio对象

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'hparams',
    'batch_size=1',
    'A comma-separated list of `name=value` hyperparameter values.')
tf.app.flags.DEFINE_string(
    'server', 'localhost:8500',
    'PredictionService host:port')
tf.app.flags.DEFINE_float(
    'frame_threshold', 0.5,
    'Threshold to use when sampling from the acoustic model.')
tf.app.flags.DEFINE_float(
    'onset_threshold', 0.5,
    'Threshold to use when sampling from the acoustic model.')
tf.app.flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')


def parse_onset_frame(onset_frame):
    hparams = tf_utils.merge_hparams(constants.DEFAULT_HPARAMS, model.get_default_hparams())
    hparams.parse(FLAGS.hparams)
    # preprocess cut consistance frames according to onset
    temp = np.ones(onset_frame[const.ONSET].shape)
    for h in range(onset_frame[const.ONSET].shape[1]):
        for l in range(onset_frame[const.ONSET].shape[0]-1):
            if onset_frame[const.ONSET][l, h]<onset_frame[const.ONSET][l+1, h]: temp[l, h]=0
    onset_frame[const.FRAME] = temp * onset_frame[const.FRAME]
    sequence_prediction = sequences_lib.pianoroll_to_note_sequence(
        onset_frame[const.FRAME],
        frames_per_second=data.hparams_frames_per_second(hparams),
        min_duration_ms=0,
        onset_predictions=onset_frame[const.ONSET],
        velocity_values=onset_frame[const.VELOCITY])
    for note in sequence_prediction.notes: note.pitch=note.pitch+constants.MIN_MIDI_PITCH
    return sequences_lib.sequence_to_pianoroll(sequence_prediction, data.hparams_frames_per_second(hparams), 21, 108).active_velocities

def buf_to_float(x, n_bytes=2, dtype=np.float32):
    """Convert an integer buffer to floating point values.
    This is primarily useful when loading integer-valued wav data
    into numpy arrays.

    See Also
    --------
    buf_to_float

    Parameters
    ----------
    x : np.ndarray [dtype=int]
        The integer-valued data buffer

    n_bytes : int [1, 2, 4]
        The number of bytes per sample in `x`

    dtype : numeric type
        The target output type (default: 32-bit float)

    Returns
    -------
    x_float : np.ndarray [dtype=float]
        The input data buffer cast to floating point
    """

    # Invert the dd of the data
    dd = 1./float(1 << ((8 * n_bytes) - 1))

    # Construct the format string
    fmt = '<i{:d}'.format(n_bytes)

    # Redd and format the data buffer
    return dd * np.frombuffer(x, fmt).astype(dtype)

class Preprocess:
    def __init__(self, hparams):
        self.hparams = hparams
        """Initializes a transcription session."""
        self.session = tf.Session()
        self.examples = tf.placeholder(tf.string, [None])
        self.batch, self.iterator = data.provide_batch(
            batch_size=1,
            examples=self.examples,
            hparams=self.hparams,
            is_training=False,
            truncated_length=0)

        self.filename = 'record/2018-11-6-test.wav'

    def create_example(self, samples):
        """Processes an audio file into an Example proto."""
        example_time = time.time()
        if self.hparams.normalize_audio:
            samples = librosa.util.normalize(samples)
        wav_data = audio_io.samples_to_wav_data(samples, self.hparams.sample_rate)

        example = tf.train.Example(features=tf.train.Features(feature={
            'id':
                tf.train.Feature(bytes_list=tf.train.BytesList(
                    value=[self.filename.encode('utf-8')]
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

    def get_spec(self, samples):
        init_time = time.time()
        self.session.run(
            self.iterator.initializer,
            feed_dict={self.examples: [self.create_example(samples)]}
            )
        print('|| init time:', time.time() - init_time)
        batch_time = time.time()
        batch = self.session.run(self.batch)
        print('|| batch time:', time.time() - batch_time)
        return batch.spec


class Transcription:
    def __init__(self):
        # create data 
        self.hparams = tf_utils.merge_hparams(constants.DEFAULT_HPARAMS, model.get_default_hparams())
        self.hparams.parse(FLAGS.hparams)
        self.checkpoint_dir = 'train2'
        # self.checkpoint_dir = '/media/admin1/32B44FF2B44FB75F/share_folder/checkpoints/new_google_single_direction_311735'

        self.preprocess = Preprocess(self.hparams)
        self.get_model()
        self.cell_state = [np.zeros([self.hparams.batch_size, self.hparams.onset_lstm_units])]*2

    def transcrib(self, samples):
        spec_time = time.time()
        spec = self.preprocess.get_spec(samples)
        print('|| spec time:', time.time() - spec_time)
        
        transcrib_time = time.time()
        frame_logits, onset_logits, velocity_values, _ = self.session.run([self.frame_probs_flat, self.onset_probs_flat, 
                                                                        self.velocity_values_flat, self.update_op],
                                                                        feed_dict={self.spec: spec})

        frame_predictions = frame_logits > FLAGS.frame_threshold
        onset_predictions = onset_logits > FLAGS.onset_threshold
        print('|| transcrib time:', time.time() - transcrib_time)
        print('------------------------shapes: ', onset_predictions.shape, frame_predictions.shape, velocity_values.shape, spec.shape,
                                                    spec[0, const.PADDING_ERROR: const.PADDING_ERROR+CHUNK_NUM, :, 0].shape)
        return onset_predictions, frame_predictions, velocity_values, spec[0, const.PADDING_ERROR: const.PADDING_ERROR+CHUNK_NUM, :, 0]
                                                            
    def get_model(self):
        with tf.Graph().as_default():
            self.spec = tf.placeholder(tf.float32, [None, None, 229, 1], 'spec_ph')
            onsets = tf.zeros([tf.shape(self.spec)[0], tf.shape(self.spec)[1]-2*const.PADDING_ERROR, 88], tf.float32, 'onsets_ph')
            velocities = tf.zeros([tf.shape(self.spec)[0], tf.shape(self.spec)[1]-2*const.PADDING_ERROR, 88], tf.float32, 'velocities_ph')
            labels = tf.zeros([tf.shape(self.spec)[0], tf.shape(self.spec)[1]-2*const.PADDING_ERROR, 88], tf.float32, 'labels_ph')
            label_weights = tf.zeros([tf.shape(self.spec)[0], tf.shape(self.spec)[1]-2*const.PADDING_ERROR, 88], tf.float32, 'lable_weights_ph')
            lengths = tf.fill((1, ), tf.shape(self.spec)[1]-2*const.PADDING_ERROR, name='lengths_ph')
            batch = {'spec':self.spec, 'onsets':onsets, 'velocities':velocities, 'labels':labels, 'label_weights':label_weights, 'lengths':lengths}
            batch = data.TranscriptionData(batch)

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
            print('------------------------ update_op', self.update_op)

    def restore_checkpoint(self, session, acoustic_checkpoint):
        var_all = tf.global_variables()
        var_to_restore = [var for var in var_all if not 'state_' in var.name]   # added variable to model
        var_to_init = [var for var in var_all if 'state_' in var.name] 
        print('------------------------ var to init', var_to_init)
        saver = tf.train.Saver(var_list=var_to_restore)
        saver.restore(session, acoustic_checkpoint)
        session.run(tf.variables_initializer(var_to_init))

class Record:
    def __init__(self):
        self.sample_size = CHUNK*(CHUNK_NUM+2*const.PADDING_ERROR)
        self.samples = np.zeros(self.sample_size)
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

        print("============================ recording")
        #开始录音
        self.frames = []
        self.i = 0

    def recording(self):
        while True: 
            print('============================ next record')
            self.i += 1
            data_stream = self.stream.read(CHUNK*CHUNK_NUM, exception_on_overflow=False)
            decode = buf_to_float(data_stream, dtype=np.float32)
            self.samples = np.append(self.samples, decode)
            self.samples = self.samples[-self.sample_size:]

            self.frames.append(data_stream)
            yield (self.i, self.samples)
   
    def save_wavfile(self, filename):
        # 停止数据流
        self.stream.stop_stream()
        self.stream.close()

        # 关闭PyAudio
        self.p.terminate()
        # 写入录音文件
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)   
        wf.writeframes(b''.join(self.frames))

class Record_analog:
    def __init__(self):
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
            time.sleep(CHUNK*CHUNK_NUM/16000)
            self.i += 1
            self.count += 1

            self.samples = np.append(self.samples, self.wav_data[(self.count)*self.sample_step: (self.count+1)*self.sample_step])
            self.samples = self.samples[-self.sample_size:]
            yield (self.i, self.samples)
    
    def full_music(self):
        return self.wav_data

class Spec_analog:
    def __init__(self):
        hparams = tf_utils.merge_hparams(
            constants.DEFAULT_HPARAMS, model.get_default_hparams())
        wav_dir = 'record/2018-11-6-test.wav'
        from trial2 import get_batch
        batch = get_batch(hparams, wav_dir)
        self.spec = batch.spec
        self.i = 0
    def get_full_spec(self):
        # self.spec[:, 4:, :, :] = 0
        # print(self.spec.shape)
        return self.spec[:, 34:54, :, :]

    def get_descret_spec(self):
        spec = []
        for i in range(self.spec.shape[1]//4):
            spec.append(self.spec[:, (i)*4:(i+1)*4, :, :])
        return spec

    def spec_gen(self):
        while True:
            self.i +=1
            if 4*self.i >= self.spec.shape[1]: 
                self.i = 1
                print('\n'*3)
                print('set i to 0')
            yield self.spec[:, (self.i-1)*4:self.i*4, :, :]

class Transcription_analog:
    def __init__(self):
        self.spec_descret = Spec_analog().get_descret_spec()
        print(len(self.spec_descret), self.spec_descret[0].shape)
        self.spec_ful = Spec_analog().get_full_spec()
        self.hparams = tf_utils.merge_hparams(constants.DEFAULT_HPARAMS, model.get_default_hparams())
        self.hparams.parse(FLAGS.hparams)
        self.checkpoint_dir = 'train'
        self.cell_state = [np.zeros([self.hparams.batch_size, self.hparams.onset_lstm_units])]*2
        self.predict()
    def predict(self):
        with tf.Graph().as_default():
            spec = tf.placeholder(tf.float32, [None, None, 229, 1], 'spec_ph')
            onsets = tf.zeros([tf.shape(spec)[0], tf.shape(spec)[1], 88], tf.float32, 'onsets_ph')
            velocities = tf.zeros([tf.shape(spec)[0], tf.shape(spec)[1], 88], tf.float32, 'velocities_ph')
            labels = tf.zeros([tf.shape(spec)[0], tf.shape(spec)[1], 88], tf.float32, 'labels_ph')
            label_weights = tf.zeros([tf.shape(spec)[0], tf.shape(spec)[1], 88], tf.float32, 'lable_weights_ph')
            lengths = tf.fill((1, ), tf.shape(spec)[1], name='lengths_ph')
            batch = {'spec':spec, 'onsets':onsets, 'velocities':velocities, 'labels':labels, 'label_weights':label_weights, 'lengths':lengths}
            batch = data.TranscriptionData(batch)

            model.get_model(batch, self.hparams, is_training=False)

            onset_probs_flat = tf.get_default_graph().get_tensor_by_name(
                'onsets/onset_probs_flat:0')
            frame_probs_flat = tf.get_default_graph().get_tensor_by_name(
                'frame_probs_flat:0')
            velocity_values_flat = tf.get_default_graph().get_tensor_by_name(
                'velocity/velocity_values_flat:0')
            init_state_op = tf.get_collection('init_state')[0]
            input_state_op = tf.get_collection('input_state')[0]
            output_state_op = tf.get_collection('output_state')[0]
            conv_output_op = tf.get_collection('conv_output')[0]
            collection = tf.get_collection('collection')
            state = [np.zeros([self.hparams.batch_size, self.hparams.onset_lstm_units])]*2

            session = tf.Session()
            saver = tf.train.Saver()
            saver.restore(session, tf.train.latest_checkpoint(self.checkpoint_dir))

            # frame_logits, onset_logits, self.velocity_values, conv_output, collection_result = session.run([frame_probs_flat, onset_probs_flat, 
            #                                                     velocity_values_flat, conv_output_op, collection],
            #                                                     feed_dict={spec: self.spec_ful, input_state_op:state})
            # self.frame_predictions = frame_logits > 0.5
            # self.onset_predictions = onset_logits > 0.5
            # print('------------------------------------------- step output')
            # np.set_printoptions(threshold=10000000000)
            # # print(collection_result[1][0, :5, :4])
            # print('------------------------------------------- conv output')
            # print(conv_output[0, :, :11])


            self.velocity_values, self.frame_predictions, self.onset_predictions = [], [], []
            for i in range(len(self.spec_descret)):
                        frame_logits, onset_logits, velocity_values, _, collection_result = session.run([frame_probs_flat, onset_probs_flat, 
                                                                        velocity_values_flat, output_state_op, collection],
                                                                        feed_dict={spec: self.spec_descret[i], input_state_op:state})
                        # print('-------------------------------------collection_result')
                        # for k in collection_result[0]:
                        #     print(type(k), k.shape)
                        #     print(k)
                        # for k in collection_result[1:]:
                        #     print(type(k), k.shape)
                        np.set_printoptions(threshold=10000000000)
                        print('------------------------------------------- step output')
                        print(collection_result[1])
                        if i == 0: continue
                        break
                        self.velocity_values.append(velocity_values)
                        self.frame_predictions.append(frame_logits > 0.5)
                        self.onset_predictions.append(onset_logits > 0.5)
            self.velocity_values = np.concatenate((self.velocity_values), axis=0)
            self.frame_predictions = np.concatenate((self.frame_predictions), axis=0)
            self.onset_predictions = np.concatenate((self.onset_predictions), axis=0)

    def plot(self):
        import matplotlib.pyplot as plt 
        plt.figure(num=1, figsize=(30, 20), dpi=100)
        pcolor = plt.pcolormesh(self.frame_predictions.T, cmap='jet', vmin=0, vmax=1)
        plt.figure(num=2, figsize=(30, 20), dpi=100)
        pcolor = plt.pcolormesh(self.onset_predictions.T, cmap='jet', vmin=0, vmax=1)
        plt.figure(num=3, figsize=(30, 20), dpi=100)
        pcolor = plt.pcolormesh(self.velocity_values.T, cmap='jet', vmin=0, vmax=1)

        sequence_prediction = sequences_lib.pianoroll_to_note_sequence(
            self.frame_predictions,
            frames_per_second=data.hparams_frames_per_second(
            self.hparams),
            min_duration_ms=0,
            onset_predictions=self.onset_predictions,
            velocity_values=self.velocity_values)

        for note in sequence_prediction.notes:
            note.pitch += constants.MIN_MIDI_PITCH

        merge = sequences_lib.sequence_to_pianoroll(sequence_prediction, data.hparams_frames_per_second(self.hparams), 21, 108).active_velocities
        plt.figure(num=4, figsize=(30, 20), dpi=100)
        pcolor = plt.pcolormesh(merge.T, cmap='jet', vmin=0, vmax=1)
        plt.show()



if __name__=='__main__':
    hparams = tf_utils.merge_hparams(constants.DEFAULT_HPARAMS, model.get_default_hparams())
    hparams.parse(FLAGS.hparams)
    pre = Preprocess(hparams)

    # test_data = np.random.rand(4608)
    # spec = pre.get_spec(test_data)
    # print('='*33)
    # print(test_data.shape)
    # print(spec.shape)

    # record =  Record()
    # a = next(record.recording())
    # print(a[0], a[1].shape)

    # record_analog = Record_analog()
    # samples = record_analog.full_music()
    # print(samples.shape)
    # spec1 = pre.get_spec(samples)
    # spec = Spec_analog().get_spec()
    # if (spec1==spec).all(): print('ksjdf[opaimnasdlighownjmvaoiphjonpg')
    # print(spec.shape)

    # import matplotlib.pyplot as plt 
    # plt.figure(num=1, figsize=(30, 20), dpi=100)
    # pcolor = plt.pcolormesh(spec[0, :, :, 0].T, cmap='jet')
    # plt.show()

    transcription_analog = Transcription_analog()
    transcription_analog.plot()
    