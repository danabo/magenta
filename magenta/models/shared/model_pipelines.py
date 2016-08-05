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

# internal imports
from magenta.pipelines import dag_pipeline
from magenta.pipelines import pipeline
from magenta.pipelines import pipelines_common
from magenta.models.shared import melody_rnn_create_dataset
from magenta.models.basic_rnn import basic_rnn_encoder_decoder
from magenta.models.lookback_rnn import lookback_rnn_encoder_decoder
from magenta.models.attention_rnn import attention_rnn_encoder_decoder


class EncoderPipeline(pipeline.Pipeline):
  """A Module that converts monophonic melodies to a model specific encoding."""

  def __init__(self, melody_encoder_decoder):
    """Constructs a EncoderPipeline.

    A melodies_lib.MelodyEncoderDecoder is needed to provide the
    `encode` function.

    Args:
      melody_encoder_decoder: A melodies_lib.MelodyEncoderDecoder object.
    """
    super(EncoderPipeline, self).__init__(
        input_type=melodies_lib.MonophonicMelody,
        output_type=tf.train.SequenceExample)
    self.melody_encoder_decoder = melody_encoder_decoder

  def transform(self, melody):
    encoded = self.melody_encoder_decoder.encode(melody)
    return [encoded]

  def get_stats(self):
    return {}

def _get_pipeline(melody_encoder_decoder, eval_ratio):
  """Returns the Pipeline instance which creates the RNN dataset.

  Args:
    melody_encoder_decoder: A melodies_lib.MelodyEncoderDecoder object.

  Returns:
    A pipeline.Pipeline instance.
  """
  quantizer = pipelines_common.Quantizer(steps_per_beat=4)
  melody_extractor = pipelines_common.MonophonicMelodyExtractor(
      min_bars=7, min_unique_pitches=5,
      gap_bars=1.0, ignore_polyphonic_notes=False)
  encoder_pipeline = EncoderPipeline(melody_encoder_decoder)
  partitioner = pipelines_common.RandomPartition(
      tf.train.SequenceExample,
      ['eval_melodies', 'training_melodies'],
      [eval_ratio])

  dag = {quantizer: dag_pipeline.Input(music_pb2.NoteSequence),
         melody_extractor: quantizer,
         encoder_pipeline: melody_extractor,
         partitioner: encoder_pipeline,
         dag_pipeline.Output(): partitioner}
  return dag_pipeline.DAGPipeline(dag)

MODEL_PIPELINES = {
    'basic_rnn': model_pipelines.basic_rnn_pipeline,
    'lookback_rnn': model_pipelines.lookback_rnn_pipeline,
    'attention_rnn': model_pipelines.attention_rnn_pipeline}

def multi_model_pipeline(models, model_pipeline_keywords={eval_ratio: 0.0}):
  instanaces = [MODEL_PIPELINES[model](**model_pipeline_keywords) for model in models]
  dag = dict([(instanace, dag_pipeline.Input(music_pb2.NoteSequence)) for instance in instances])
  dag.update(dict([(dag_pipeline.Output(), instance) for instance in instances]))

def basic_rnn_pipeline(eval_ratio=0.0):
  _get_pipeline(basic_rnn_encoder_decoder.MelodyEncoderDecoder(), eval_ratio)

def lookback_rnn_pipeline(eval_ratio=0.0):
  _get_pipeline(lookback_rnn_encoder_decoder.MelodyEncoderDecoder(), eval_ratio)

def attention_rnn_pipeline(eval_ratio=0.0):
  _get_pipeline(attention_rnn_encoder_decoder.MelodyEncoderDecoder(), eval_ratio)
