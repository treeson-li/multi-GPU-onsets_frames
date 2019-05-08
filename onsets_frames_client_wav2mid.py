import collections
import os

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
import numpy as np

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


def create_example(filename, hparams):
    """Processes an audio file into an Example proto."""
    wav_data = audio_io.samples_to_wav_data(    # class bytes
        librosa.util.normalize(librosa.core.load(
            filename, sr=hparams.sample_rate)[0]), hparams.sample_rate)

    example = tf.train.Example(features=tf.train.Features(feature={
        'id':
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[filename.encode('utf-8')]
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
    return example.SerializeToString()

def get_spec(hparams, filename):
    """Initializes a transcription session."""
    with tf.Session() as session:
        examples = tf.placeholder(tf.string, [None])
        batch, iterator = data.provide_batch(
            batch_size=1,
            examples=examples,
            hparams=hparams,
            is_training=False,
            truncated_length=0)

        session.run(
            iterator.initializer,
            feed_dict={examples: [create_example(filename, hparams)]}
            )
        batch = session.run(batch)
        return batch.spec

def main(wav_dir):
    # create data 
    hparams = tf_utils.merge_hparams(constants.DEFAULT_HPARAMS, model.get_default_hparams())
    hparams.parse(FLAGS.hparams)
    spec = get_spec(hparams, wav_dir)

    # create the RPC stub
    channel = grpc.insecure_channel(FLAGS.server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # create the request object and set the name and signature_name params
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'onsets_frames'
    request.model_spec.signature_name = 'predict_results'

    # fill in the request object with the necessary data
    request.inputs['spec'].CopyFrom(tf.contrib.util.make_tensor_proto(spec, shape=spec.shape))

    result_future = stub.Predict.future(request, 20.0)  # 5 seconds
    shapex = result_future.result().outputs['onset'].tensor_shape.dim[0].size
    shapey = result_future.result().outputs['onset'].tensor_shape.dim[1].size
    onset = np.array(result_future.result().outputs['onset'].float_val).reshape([shapex, shapey])
    frame = np.array(result_future.result().outputs['frame'].float_val).reshape([shapex, shapey])
    velocity = np.array(result_future.result().outputs['velocity'].float_val).reshape([shapex, shapey])
    # print('*'*8, shapex, shapey)
    # print(result_future.result().outputs['onset'].tensor_shape)
    # print(type(velocity), velocity.shape, onset.shape, frame.shape)
    # print(onset, frame, velocity)

    frame_predictions = frame > FLAGS.frame_threshold
    onset_predictions = onset > FLAGS.onset_threshold
    sequence_prediction = sequences_lib.pianoroll_to_note_sequence(
        frame_predictions,
        frames_per_second=data.hparams_frames_per_second(hparams),
        min_duration_ms=0,
        onset_predictions=onset_predictions,
        velocity_values=velocity)
    for note in sequence_prediction.notes: note.pitch=note.pitch+21
    # print(sequence_prediction)
    midi_filename = '/tmp/server.midi'
    midi_io.sequence_proto_to_midi_file(sequence_prediction, midi_filename)
    print('Transcription written to %s.', midi_filename)


if __name__=='__main__':
    wav_dir = 'record/2018-11-6-test.wav'
    # wav_dir = 'record/changba_real_sound_hold_the_mic.wav'
    # wav_dir = 'record/high8.wav'
    # wav_dir = 'record/low8.wav'
    # wav_dir = 'record/low16.wav'
    # wav_dir = 'record/long.wav'
    # wav_dir = 'record/changba_real_sound_on_the_piano.wav'    main(checkpoint_dir)
    main(wav_dir)
