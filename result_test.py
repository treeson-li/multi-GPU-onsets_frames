import collections
import os

import librosa
import tensorflow as tf
import constants
import data
import model
import numpy as np

from magenta.common import tf_utils
from magenta.music import audio_io
from magenta.music import midi_io
from magenta.music import sequences_lib
from magenta.protobuf import music_pb2
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

TranscriptionSession = collections.namedtuple(
    'TranscriptionSession',
    ('session', 'examples', 'iterator', 'onset_probs_flat', 'frame_probs_flat',
     'velocity_values_flat', 'hparams'))

def create_example(filename, hparams):
    """Processes an audio file into an Example proto."""
    wav_data = librosa.core.load(filename, sr=hparams.sample_rate)[0]
    if hparams.normalize_audio:
        wav_data = librosa.util.normalize(wav_data)
    wav_data = audio_io.samples_to_wav_data(wav_data, hparams.sample_rate)

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
    print('--------------------------------------------------example')
    print(type(example))
    # print(example.SerializeToString())
    return example.SerializeToString()

def restore_checkpoint(session, acoustic_checkpoint):
    var_all = tf.global_variables()
    var_to_restore = [var for var in var_all if not 'state_' in var.name]
    var_to_init = [var for var in var_all if 'state_' in var.name] 
    # print('#'*33, var_all[0].name)
    # print('#'*33, var_all)
    print('#'*33, var_to_init)
    saver = tf.train.Saver(var_list=var_to_restore)
    saver.restore(session, acoustic_checkpoint)
    session.run(tf.variables_initializer(var_to_init))


def initialize_session(acoustic_checkpoint, hparams):
    """Initializes a transcription session."""
    with tf.Graph().as_default():
        examples = tf.placeholder(tf.string, [None])

        batch, iterator = data.provide_batch(
            batch_size=1,
            examples=examples,
            hparams=hparams,
            is_training=False,
            truncated_length=0)
        print('----------------------------------------------------batch, iterator')
        print(type(batch), batch)
        print(iterator)
        model.get_model(batch, hparams, is_training=False)

        session = tf.Session()
        restore_checkpoint(session, acoustic_checkpoint)
        
        onset_probs_flat = tf.get_default_graph().get_tensor_by_name(
            'onsets/onset_probs_flat:0')
        frame_probs_flat = tf.get_default_graph().get_tensor_by_name(
            'frame_probs_flat:0')
        velocity_values_flat = tf.get_default_graph().get_tensor_by_name(
            'velocity/velocity_values_flat:0')

        return TranscriptionSession(
            session=session,
            examples=examples,
            iterator=iterator,
            onset_probs_flat=onset_probs_flat,
            frame_probs_flat=frame_probs_flat,
            velocity_values_flat=velocity_values_flat,
            hparams=hparams)

def transcribe_audio(transcription_session, filename, frame_threshold,
                     onset_threshold):
    """Transcribes an audio file."""
    tf.logging.info('Processing file...')
    transcription_session.session.run(
        transcription_session.iterator.initializer,
        feed_dict={transcription_session.examples: [create_example(filename, transcription_session.hparams)]}
        )

    tf.logging.info('Running inference...')
    frame_logits, onset_logits, velocity_values = transcription_session.session.run([
                                                                transcription_session.frame_probs_flat,
                                                                transcription_session.onset_probs_flat,
                                                                transcription_session.velocity_values_flat])

    frame_predictions = frame_logits > frame_threshold
    onset_predictions = onset_logits > onset_threshold

    import matplotlib.pyplot as plt 
    # plt.figure(num=1, figsize=(30, 20), dpi=100)
    # pcolor = plt.pcolormesh(frame_predictions.T, cmap='jet', vmin=0, vmax=1)
    plt.figure(num=2, figsize=(30, 20), dpi=100)
    pcolor = plt.pcolormesh(onset_predictions.T, cmap='jet', vmin=0, vmax=1)
    # plt.figure(num=3, figsize=(30, 20), dpi=100)
    # pcolor = plt.pcolormesh(velocity_values.T, cmap='jet', vmin=0, vmax=1)

    sequence_prediction = sequences_lib.pianoroll_to_note_sequence(
        frame_predictions,
        frames_per_second=data.hparams_frames_per_second(
            transcription_session.hparams),
        min_duration_ms=0,
        onset_predictions=onset_predictions,
        velocity_values=velocity_values)

    for note in sequence_prediction.notes:
        note.pitch += constants.MIN_MIDI_PITCH

    merge = sequences_lib.sequence_to_pianoroll(sequence_prediction, data.hparams_frames_per_second(transcription_session.hparams), 21, 108).active_velocities
    # import pickle
    # pickle.dump(merge, open('/tmp/pickle_file', 'wb'))
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', type(merge), merge.shape)
    plt.figure(num=4, figsize=(30, 20), dpi=100)
    pcolor = plt.pcolormesh(merge.T, cmap='jet', vmin=0, vmax=1)
    plt.show()
   
    return sequence_prediction


def main(wav_dir, checkpoint_dir):
    frame_threshold = 0.5
    onset_threshold = 0.5
    acoustic_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    print('=========================================================================checkpoint_dir, acoustic_checkpoint')
    print(checkpoint_dir)
    print(acoustic_checkpoint)

    hparams = tf_utils.merge_hparams(
    constants.DEFAULT_HPARAMS, model.get_default_hparams())
    hparams.batch_size = 1

    transcription_session = initialize_session(acoustic_checkpoint, hparams)
    print('Starting transcription for %s...', wav_dir)
    sequence_prediction = transcribe_audio(
        transcription_session, wav_dir, frame_threshold, onset_threshold)

    midi_filename = os.path.splitext(wav_dir)[0] + '.midi'
    midi_io.sequence_proto_to_midi_file(sequence_prediction, midi_filename)
    print('Transcription written to %s.', midi_filename)


if __name__=='__main__':
    wav_dir = 'record/2018-11-6-test.wav'
    # wav_dir = '/media/admin1/32B44FF2B44FB75F/Data/maestro/maestro-v1.0.0/2017/MIDI-Unprocessed_044_PIANO044_MID--AUDIO-split_07-06-17_Piano-e_1-04_wav--2.wav'
    # wav_dir = '/home/admin1/sz/onset_frame/magenta/magenta/models/onsets_frames_transcription/output.wav'
    # wav_dir = 'record/changba_real_sound_hold_the_mic.wav'
    # wav_dir = 'record/high8.wav'
    # wav_dir = 'record/low8.wav'
    # wav_dir = 'record/low16.wav'
    # wav_dir = 'record/long.wav'
    # wav_dir = 'record/changba_real_sound_on_the_piano.wav'
    # wav_dir = 'record/1114.wav'
    
    checkpoint_dir = 'train2'
    checkpoint_dir = '/media/admin1/32B44FF2B44FB75F/share_folder/checkpoints/new_google_single_direction_311735'
    # checkpoint_dir = '/home/admin1/sz/onset_frame/magenta/magenta/models/onsets_frames_transcription/mymodel/basic'
    # checkpoint_dir = 'mymodel/train'
    main(wav_dir, checkpoint_dir)
