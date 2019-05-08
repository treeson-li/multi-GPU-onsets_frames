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

import grpc
from grpc.beta import implementations
import predict_pb2
import prediction_service_pb2_grpc
import requests
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 定义数据流块
# CHUNK = 1024

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1600    # CHUNK should divide RATE evenly
# 录音时间
TRANSCRIPTION_SECONDS = 20
RECORD_SECONDS = 500
# 要写入的文件名
WAVE_OUTPUT_FILENAME = "output.wav"
MIDI_OUTPUT_FILENAME = "output.mid"
# 创建PyAudio对象

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'server', 'localhost:8500',
    'PredictionService host:port')
tf.app.flags.DEFINE_string(
    'hparams',
    'onset_mode=length_ms,onset_length=32',
    'A comma-separated list of `name=value` hyperparameter values.')
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

        self.filename = 'whaterver.wav'

    def create_example(self, samples):
        """Processes an audio file into an Example proto."""
        # wav_data = audio_io.samples_to_wav_data(
        #     librosa.util.normalize(
        #         scipy.io.wavfile.read(filename)), hparams.sample_rate)
        example_time = time.time()
        wav_data = audio_io.samples_to_wav_data(
            librosa.util.normalize(samples), RATE)
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
        print('example time:', time.time() - example_time)
        return example.SerializeToString()

    def get_spec(self, samples):
        session_time = time.time()
        self.session.run(
            self.iterator.initializer,
            feed_dict={self.examples: [self.create_example(samples)]}
            )
        batch = self.session.run(self.batch)
        print('session time:', time.time() - session_time)
        return batch.spec

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

    # Invert the scale of the data
    scale = 1./float(1 << ((8 * n_bytes) - 1))

    # Construct the format string
    fmt = '<i{:d}'.format(n_bytes)

    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)

class Transcription:
    def __init__(self):
        # create data 
        self.hparams = tf_utils.merge_hparams(constants.DEFAULT_HPARAMS, model.get_default_hparams())
        self.hparams.parse(FLAGS.hparams)

        # create the RPC stub
        channel = grpc.insecure_channel(FLAGS.server)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        # create the request object and set the name and signature_name params
        self.request = predict_pb2.PredictRequest()
        self.request.model_spec.name = 'onsets_frames'
        self.request.model_spec.signature_name = 'predict_results'

        self.preprocess = Preprocess(self.hparams)
        
        # fill in the request object with the necessary data
    def transcrib(self, samples):
        spec_time = time.time()
        spec = self.preprocess.get_spec(samples)
        print('spec time:', time.time() - spec_time)
        request_time = time.time()
        self.request.inputs['spec'].CopyFrom(tf.contrib.util.make_tensor_proto(spec, shape=spec.shape))

        result_future = self.stub.Predict.future(self.request, 20.0)  # 5 seconds
        shapex = result_future.result().outputs['onset'].tensor_shape.dim[0].size
        shapey = result_future.result().outputs['onset'].tensor_shape.dim[1].size
        onset = np.array(result_future.result().outputs['onset'].float_val).reshape([shapex, shapey])
        frame = np.array(result_future.result().outputs['frame'].float_val).reshape([shapex, shapey])
        velocity = np.array(result_future.result().outputs['velocity'].float_val).reshape([shapex, shapey])
        print('request_time:', time.time() - request_time)

        frame_predictions = frame > FLAGS.frame_threshold
        onset_predictions = onset > FLAGS.onset_threshold
        sequence_prediction = sequences_lib.pianoroll_to_note_sequence(
            frame_predictions,
            frames_per_second=data.hparams_frames_per_second(self.hparams),
            min_duration_ms=0,
            onset_predictions=onset_predictions,
            velocity_values=velocity)
        for note in sequence_prediction.notes: note.pitch=note.pitch+constants.MIN_MIDI_PITCH
        # print(sequence_prediction)
        return sequences_lib.sequence_to_pianoroll(sequence_prediction, data.hparams_frames_per_second(self.hparams), 21, 108)
        # midi_filename = six.BytesIO()
        # midi_io.sequence_proto_to_midi_file(sequence_prediction, midi_filename)
        # print('Transcription written to %s.', midi_filename)
        # return midi_filename

class Record:
    def __init__(self):
        # np.set_printoptions(threshold=10000000000)
        # self.transcription = Transcription()
        self.samples = np.zeros(TRANSCRIPTION_SECONDS*RATE)
        # self.transcription.transcrib(self.samples)  # warm it

        self.p = pyaudio.PyAudio()
        print("============================* recording")
        self.stream = self.p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
        #开始录音
        self.frames = []
        self.i = 0
        print('============================* init done')

    def recording(self):
        print('============================* next record')
        # while self.i < int(RATE / CHUNK * RECORD_SECONDS):
        while True:
            self.i += 1
            data_stream = self.stream.read(CHUNK, exception_on_overflow=False)
            decode = buf_to_float(data_stream, dtype=np.float32)
            self.samples = np.append(self.samples, decode)
            self.samples = self.samples[-TRANSCRIPTION_SECONDS*RATE:]
            
            self.frames.append(data_stream)
            transcrib_interval = 5
            if self.i % transcrib_interval == 0:
                yield (self.i // transcrib_interval, self.samples)
            # if ((self.i+1)*CHUNK) % (RATE) == 0:
            #     # print('='*44)
            #     transcrib_time = time.time()
            #     piano_roll = self.transcription.transcrib(self.samples)
            #     print('transcrib_time:', time.time() - transcrib_time)

            #     try:
            #         queue.put(piano_roll.active_velocities.T)
            #         print('size', queue.qsize())
            #     except queue.Full:
            #         print('queue is full')
            #         break
   
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


if __name__=='__main__':
    record = Record()
    import time
    for i in range(100):
        time.sleep(1)
        next_record = next(record.recording())
        print('='*44)
        print(next_record)
        print(next_record[1].shape)