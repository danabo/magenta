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

import random

# internal imports
import numpy as np
import tensorflow as tf

from magenta.lib import melodies_lib
from magenta.pipelines import pipeline
from magenta.pipelines import pipeline_unit
from magenta.pipelines import pipeline_units_common
from magenta.protobuf import music_pb2


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input', None,
                           'TFRecord to read NoteSequence protos from.')
tf.app.flags.DEFINE_string('output_dir', None,
                           'Directory to write training and eval TFRecord '
                           'files. The TFRecord files are populated with '
                           'SequenceExample protos.')
tf.app.flags.DEFINE_float('eval_ratio', 0.0,
                          'Fraction of input to set aside for eval set. '
                          'Partition is randomly selected.')


class EncoderUnit(pipeline_unit.PipelineUnit):
  """A Module that converts monophonic melodies into basic_rnn samples."""
  input_type = melodies_lib.MonophonicMelody
  output_type = tf.train.SequenceExample

  def __init__(self, melody_encoder_decoder):
    """Constructor takes settings for the OneHotEncoder module.

    Args:
      melody_encoder_decoder: A melodies_lib.MelodyEncoderDecoder object.
    """
    super(EncoderUnit, self).__init__()
    self.melody_encoder_decoder = melody_encoder_decoder

  def transform(self, melody):
    encoded = self.melody_encoder_decoder.encode(melody)
    return [encoded]


class BarEncoderUnit(pipeline_unit.PipelineUnit):

  def __init__(self, melody_encoder_decoder, steps_per_bar=16):
    super(BarEncoderUnit, self).__init__()
    self.melody_encoder_decoder = melody_encoder_decoder
    self.steps_per_bar = steps_per_bar

  def transform(self, melody):
    if melody.steps_per_bar != self.steps_per_bar:
      return []
    melody.squash(self.melody_encoder_decoder.min_note, self.melody_encoder_decoder.max_note, self.melody_encoder_decoder.transpose_to_key)
    outputs = []
    for index in range(0, len(melody.events), self.steps_per_bar):
      if len(melody.events) - index <= self.steps_per_bar:
        break
      values = np.zeros((self.steps_per_bar, self.melody_encoder_decoder.input_size))
      for j in range(self.steps_per_bar):
        melody_event = melody.events[index + j]
        values[j, self.melody_encoder_decoder.melody_event_to_model_event(melody_event)] = 1.0
      feature = tf.train.Feature(float_list=tf.train.FloatList(value=values.reshape((-1,))))
      example = tf.train.Example(features=tf.train.Features(feature={'notes': feature}))
      outputs.append(example)
    return outputs


class PhraseEncoderUnit(pipeline_unit.PipelineUnit):

  def __init__(self, melody_encoder_decoder, steps_per_bar=16, phrase_bar_len=16):
    self.melody_encoder_decoder = melody_encoder_decoder
    self.steps_per_bar = steps_per_bar
    self.phrase_len = phrase_bar_len * steps_per_bar

  def transform(self, melody):
    if melody.steps_per_bar != self.steps_per_bar:
      return []
    melody.squash(self.melody_encoder_decoder.min_note, self.melody_encoder_decoder.max_note, self.melody_encoder_decoder.transpose_to_key)
    outputs = []
    for index in range(0, len(melody.events), self.phrase_len):
      if len(melody.events) - index <= self.phrase_len:
        break
      values = np.zeros((self.phrase_len,), dtype=int)
      for j in range(self.phrase_len):
        melody_event = melody.events[index + j]
        values[j] = self.melody_encoder_decoder.melody_event_to_model_event(melody_event)
      feature = tf.train.Feature(int64_list=tf.train.Int64List(value=values.reshape((-1,))))
      example = tf.train.Example(features=tf.train.Features(feature={'phrase': feature}))
      outputs.append(example)
    return outputs


def random_partition(input_list, partition_ratio):
  partitions = [], []
  for item in input_list:
    partition_index = int(random.random() < partition_ratio)
    partitions[partition_index].append(item)
  return partitions  # old, new


def map_and_flatten(input_list, func):
  return [output
          for single_input in input_list
          for output in func(single_input)]


class BasicRNNPipeline(pipeline.Pipeline):
  """A custom Pipeline implementation.

  Converts music_pb2.NoteSequence into tf.train.SequenceExample protos for use
  in the basic_rnn model.
  """
  input_type = music_pb2.NoteSequence
  output_type = tf.train.SequenceExample

  def __init__(self, melody_encoder_decoder, eval_ratio):
    super(BasicRNNPipeline, self).__init__()
    self.output_names = ['training_melodies', 'eval_melodies']
    self.eval_ratio = eval_ratio
    self.quantizer = pipeline_units_common.Quantizer(steps_per_beat=4)
    self.melody_extractor = pipeline_units_common.MonophonicMelodyExtractor(
        min_bars=7, min_unique_pitches=5,
        gap_bars=1.0, ignore_polyphonic_notes=False)
    #self.encoder_unit = BarEncoderUnit(melody_encoder_decoder)
    self.encoder_unit = PhraseEncoderUnit(melody_encoder_decoder)
    self.stats_dict = {}

  def transform(self, note_sequence):
    intermediate_objects = self.quantizer.transform(note_sequence)
    intermediate_objects = map_and_flatten(intermediate_objects,
                                           self.melody_extractor.transform)
    outputs = map_and_flatten(intermediate_objects, self.encoder_unit.transform)
    train_set, eval_set = random_partition(outputs, self.eval_ratio)

    return {self.output_names[0]: train_set, self.output_names[1]: eval_set}

  def get_stats(self):
    return {}

  def get_output_names(self):
    return self.output_names


def run_from_flags(melody_encoder_decoder):
  pipeline_instance = BasicRNNPipeline(
      melody_encoder_decoder, FLAGS.eval_ratio)
  pipeline.run_pipeline_serial(
      pipeline_instance,
      pipeline.tf_record_iterator(FLAGS.input, pipeline_instance.input_type),
      FLAGS.output_dir)
