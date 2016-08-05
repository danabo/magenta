# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Create a dataset of SequenceExamples from NoteSequence protos.

This script will extract melodies from NoteSequence protos and save them to
TensorFlow's SequenceExample protos for input to the melody RNN models.
"""

# internal imports
import tensorflow as tf

from magenta.lib import melodies_lib
from magenta.pipelines import dag_pipeline
from magenta.pipelines import pipeline
from magenta.pipelines import pipelines_common
from magenta.protobuf import music_pb2


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('midi_dir', None,
                           'Directory containing MIDI files to convert.')
tf.app.flags.DEFINE_string('note_sequence_input', None,
                           'TFRecord to read NoteSequence protos from.')
tf.app.flags.DEFINE_string('models', '',
                           'A comma seperated list of model names. Training (and eval if eval_ratio is not 0) datasets for each model in the list will be generated simultaneously. If no models are given, NoteSequnce TFRecords will be produced.')
tf.app.flags.DEFINE_string('output_dir', None,
                           'Directory to write training and eval TFRecord '
                           'files. The TFRecord files are populated with '
                           'SequenceExample protos.')
tf.app.flags.DEFINE_float('eval_ratio', 0.0,
                          'Fraction of input to set aside for eval set. '
                          'Partition is randomly selected.')


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  models = [model.strip() for model in FLAGS.models.split(',')]
  tf.logging.info('Generating datasets for these models: %s', models)

  note_sequence_to_output = multi_model_pipeline(models, {'eval_ratio': FLAGS.eval_ratio})

  if FLAGS.midi_dir:
    midi_to_note_sequence = pipelines_common.MidiToNoteSequence()
    dag = {midi_to_note_sequence: dag_pipeline.Input(midi_to_note_sequence.input_type),
           note_sequence_to_output: midi_to_note_sequence,
           dag_pipeline.Output(): note_sequence_to_output}
    master_pipeline = dag_pipeline.DAGPipeline(dag, 'Midi')
    pipeline.run_pipeline_serial(
      note_sequence_to_output,
      pipeline.file_iterator(FLAGS.midi_dir, master_pipeline.input_type),
      FLAGS.output_dir)
  elif FLAGS.note_sequence_input:
    pipeline.run_pipeline_serial(
      note_sequence_to_output,
      pipeline.tf_record_iterator(FLAGS.note_sequence_input, note_sequence_to_output.input_type),
      FLAGS.output_dir)
  else:
    raise ValueError()


if __name__ == '__main__':
  tf.app.run()
