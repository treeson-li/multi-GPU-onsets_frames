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

r"""Train Onsets and Frames piano transcription model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import train_util
import model
import constants
from magenta.common import tf_utils
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("ps_hosts", "localhost:2222", "ps hosts")
tf.app.flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224,localhost:2225,localhost:2226", "worker hosts")
tf.app.flags.DEFINE_string("job_name", "worker", "'ps' or'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

tf.app.flags.DEFINE_string('master', '',
                           'Name of the TensorFlow runtime to use.')
tf.app.flags.DEFINE_string(
    'examples_path', '/home/liming/data/tfrecord/maps/maps_config2_train_spec.tfrecord',
    'Path to a TFRecord file of train/eval examples.')
tf.app.flags.DEFINE_string(
    'run_dir', '~/data/score_template4',#score_cxt128dir1',#score_cxt128label/',#'~/data/score_cxt128head1/',#'~/data/score_cxt128merged/',#'~/data/score_cxt128/',#
    'Path where checkpoints and summary events will be located during '
    'training and evaluation. Separate subdirectories `train` and `eval` '
    'will be created within this directory.')
tf.app.flags.DEFINE_string(
    'eval_dir', None,
    'Path where eval summaries will be written. If not specified, will be a '
    'subdirectory of run_dir.')
tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'Path to the checkpoint to use in `test` mode. If not provided, latest '
    'in `run_dir` will be used.')
tf.app.flags.DEFINE_integer('num_steps', 1400000,
                            'Number of training steps or `None` for infinite.')
tf.app.flags.DEFINE_integer(
    'eval_num_batches', None,
    'Number of batches to use during evaluation or `None` for all batches '
    'in the data source.')
tf.app.flags.DEFINE_integer(
    'checkpoints_to_keep', 5,
    'Maximum number of checkpoints to keep in `train` mode or 0 for infinite.')
tf.app.flags.DEFINE_enum('mode', 'train', ['train', 'eval', 'test'],
                         'Which mode to use.')
tf.app.flags.DEFINE_string(
    'hparams', '',
    'A comma-separated list of `name=value` hyperparameter values.')
tf.app.flags.DEFINE_integer('ps_task', 0, 'The task number for this worker.')
tf.app.flags.DEFINE_integer('num_ps_tasks', 3,
                            'The number of parameter server tasks')
tf.app.flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')


def run(hparams, run_dir):
  """Run train/eval/test."""
  print('--------------'*5)
  train_dir = os.path.join(run_dir, 'train')
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")  
  # create the cluster configured by `ps_hosts' and 'worker_hosts'
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
  print('cluster', cluster)
  server=tf.train.Server(cluster,job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)
  print(FLAGS.job_name, FLAGS.task_index)
  if FLAGS.job_name == "ps":
    server.join()  # ps hosts only join
  elif FLAGS.job_name == "worker":
    if FLAGS.mode == 'eval':
      eval_dir = os.path.join(run_dir, 'eval')
      if FLAGS.eval_dir:
        eval_dir = os.path.join(eval_dir, FLAGS.eval_dir)
      train_util.evaluate(
          train_dir=train_dir,
          eval_dir=eval_dir,
          examples_path=FLAGS.examples_path,
          num_batches=FLAGS.eval_num_batches,
          hparams=hparams,
          master=FLAGS.master)
    elif FLAGS.mode == 'test':
      checkpoint_path = tf.train.latest_checkpoint(train_dir)
      if FLAGS.checkpoint_path:
        checkpoint_path = os.path.expanduser(FLAGS.checkpoint_path)
  
      tf.logging.info('Testing with checkpoint: %s', checkpoint_path)
      test_dir = os.path.join(run_dir, 'test')
      train_util.test(
          checkpoint_path=checkpoint_path,
          test_dir=test_dir,
          examples_path=FLAGS.examples_path,
          num_batches=FLAGS.eval_num_batches,
          hparams=hparams,
          master=FLAGS.master)
    elif FLAGS.mode == 'train':
      train_util.train(
          train_dir=train_dir,
          examples_path=FLAGS.examples_path,
          hparams=hparams,
          checkpoints_to_keep=FLAGS.checkpoints_to_keep,
          num_steps=FLAGS.num_steps,
          master=server.target,
          task_index=FLAGS.task_index,
          cluster=cluster)


def main(unused_argv):
  tf.logging.set_verbosity(FLAGS.log)
  tf.app.flags.mark_flags_as_required(['examples_path'])

  run_dir = os.path.expanduser(FLAGS.run_dir)

  hparams = tf_utils.merge_hparams(constants.DEFAULT_HPARAMS,
                                   model.get_default_hparams())

  # Command line flags override any of the preceding hyperparameter values.
  hparams.parse(FLAGS.hparams)

  run(hparams, run_dir)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
