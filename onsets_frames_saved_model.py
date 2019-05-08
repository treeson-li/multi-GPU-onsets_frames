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

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer(
    'model_version', 2, 
    'version number of the model.')
tf.app.flags.DEFINE_string(
    'export_model_dir', "./versions",
    'Directory where the model exported files should be placed.')
tf.app.flags.DEFINE_string(
    'acoustic_run_dir', None,
    'Path to look for acoustic checkpoints. Should contain subdir `train`.')
tf.app.flags.DEFINE_string(
    'acoustic_checkpoint_dir', 'train',
    'Filename of the checkpoint to use. If not specified, will use the latest '
    'checkpoint')
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

TranscriptionSession = collections.namedtuple(
    'TranscriptionSession',
    ('session', 'examples', 'iterator', 'onset_probs_flat', 'frame_probs_flat',
     'velocity_values_flat', 'hparams'))


def initialize_session(acoustic_checkpoint, hparams):
    """Initializes a transcription session."""
    with tf.Session(graph=tf.Graph()) as sess:
        # examples = tf.placeholder(tf.string, [None])

        # batch, iterator = data.provide_batch(
        #     batch_size=1,
        #     examples=examples,
        #     hparams=hparams,
        #     is_training=False,
        #     truncated_length=0)

        spec = tf.placeholder(tf.float32, [None, None, 229, 1], 'spec_ph')
        # onsets = tf.placeholder(tf.float32, [None, None, 88], 'onsets_ph')
        # velocities = tf.placeholder(tf.float32, [None, None, 88], 'velocities_ph')
        # labels = tf.placeholder(tf.float32, [None, None, 88], 'labels_ph')
        # label_weights = tf.placeholder(tf.float32, [None, None, 88], 'lable_weights_ph')
        # lengths = tf.placeholder(tf.int32, [None, ], 'lengths_ph')
        onsets = tf.zeros([tf.shape(spec)[0], tf.shape(spec)[1], 88], tf.float32, 'onsets_ph')
        velocities = tf.zeros([tf.shape(spec)[0], tf.shape(spec)[1], 88], tf.float32, 'velocities_ph')
        labels = tf.zeros([tf.shape(spec)[0], tf.shape(spec)[1], 88], tf.float32, 'labels_ph')
        label_weights = tf.zeros([tf.shape(spec)[0], tf.shape(spec)[1], 88], tf.float32, 'lable_weights_ph')
        # lengths = tf.Variable([tf.shape(spec)[0],], dtype=tf.int32, name='lengths_ph')
        # lengths = tf.constant(tf.shape(spec)[1], dtype=tf.int32, name='lengths_ph')
        lengths = tf.fill((1, ), tf.shape(spec)[1], name='lengths_ph')
        
        batch = {'spec':spec, 'onsets':onsets, 'velocities':velocities, 'labels':labels, 'label_weights':label_weights, 'lengths':lengths}
        batch = data.TranscriptionData(batch)

        model.get_model(batch, hparams, is_training=False)
        saver = tf.train.Saver()
        saver.restore(sess, acoustic_checkpoint)

        onset_probs_flat = tf.get_default_graph().get_tensor_by_name(
            'onsets/onset_probs_flat:0')
        frame_probs_flat = tf.get_default_graph().get_tensor_by_name(
            'frame_probs_flat:0')
        velocity_values_flat = tf.get_default_graph().get_tensor_by_name(
            'velocity/velocity_values_flat:0')

        # Export model
        # WARNING(break-tutorial-inline-code): The following code snippet is
        # in-lined in tutorials, please update tutorial documents accordingly
        # whenever code changes.
        export_path_base = FLAGS.export_model_dir
        export_path = os.path.join(
            tf.compat.as_bytes(export_path_base),
            tf.compat.as_bytes(str(FLAGS.model_version)))
        print('Exporting trained model to', export_path)
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)

        # Build the signature_def_map.
        # Creates the TensorInfo protobuf objects that encapsulates the input/output tensors
        tensor_info_spec = tf.saved_model.utils.build_tensor_info(spec)
        tensor_info_onset = tf.saved_model.utils.build_tensor_info(onset_probs_flat)
        tensor_info_frame = tf.saved_model.utils.build_tensor_info(frame_probs_flat)
        tensor_info_velocity = tf.saved_model.utils.build_tensor_info(velocity_values_flat)

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'spec': tensor_info_spec},
                outputs={'onset': tensor_info_onset, 'frame':tensor_info_frame, 'velocity':tensor_info_velocity},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(
            sess, 
            [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_results':
                    prediction_signature,
            },
            main_op=tf.tables_initializer(),
            strip_default_attrs=True)

        builder.save()
        print('Done exporting!')


def main(checkpoint_dir):
    acoustic_checkpoint = tf.train.latest_checkpoint(FLAGS.acoustic_checkpoint_dir)
    hparams = tf_utils.merge_hparams(constants.DEFAULT_HPARAMS, model.get_default_hparams())
    hparams.parse(FLAGS.hparams)
    initialize_session(acoustic_checkpoint, hparams)

if __name__=='__main__':
    checkpoint_dir = 'train'
    main(checkpoint_dir)
