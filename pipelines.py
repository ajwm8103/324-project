from note_seq import *
import copy, numbers, torch, pickle
from tqdm import TqdmSynchronisationWarning

class EncoderPipeline():
  """A Pipeline that converts performances to a model specific encoding."""

  def __init__(self, encoder_decoder, control_signals, optional_conditioning):
    """Constructs an EncoderPipeline.

    Args:
      config: A PerformanceRnnConfig that specifies the encoder/decoder and
          note density conditioning behavior.
      name: A unique pipeline name.
    """
    #super(EncoderPipeline, self).__init__(
    #    input_type=note_seq.BasePerformance,
    #    output_type=tf.train.SequenceExample,
    #    name=name)
    self._encoder_decoder = encoder_decoder
    self._control_signals = control_signals
    self._optional_conditioning = optional_conditioning

  def transform(self, input_object):
    performance = input_object

    if self._control_signals:
      # Encode conditional on control signals.
      control_sequences = []
      for control in self._control_signals:
        control_sequences.append(control.extract(performance))
      control_sequence = list(zip(*control_sequences))
      if self._optional_conditioning:
        # Create two copies, one with and one without conditioning.
        # pylint: disable=g-complex-comprehension
        encoded = [
            self._encoder_decoder.encode(
                list(zip([disable] * len(control_sequence), control_sequence)),
                performance) for disable in [False, True]
        ]
        # pylint: enable=g-complex-comprehension
      else:
        encoded = [self._encoder_decoder.encode(
            control_sequence, performance)]
    else:
      # Encode unconditional.
      input_data, label_data = self._encoder_decoder.encode(performance)
      a = [input_data[0][0]]
      a += label_data
      
      encoded = torch.tensor(a)
    return encoded

def make_sequence_example(inputs, labels):
  """Returns a SequenceExample for the given inputs and labels.

  Args:
    inputs: A list of input vectors. Each input vector is a list of floats.
    labels: A list of ints.

  Returns:
    A tf.train.SequenceExample containing inputs and labels.
  """
  #print('sussy', len(inputs), labels)
  input_features = [torch.tensor(input_, dtype=torch.int64) for input_ in inputs]
  label_features = torch.tensor(labels, dtype=torch.int64)
  #for label in labels:
  #  if isinstance(label, numbers.Number):
  #    label = [label]
  #  label_features.append(torch.)
  feature_list = {
      'inputs': input_features,
      'labels': label_features,
  }
  return feature_list

class PerformanceExtractor():
    """Extracts polyphonic tracks from a quantized NoteSequence."""

    def __init__(self, min_events, max_events, num_velocity_bins,
            note_performance, name=None):
        #super(PerformanceExtractor, self).__init__(
        #    input_type=note_seq.music_pb2.NoteSequence,
        #    output_type=BasePerformance,
        #    name=name)
        self._min_events = min_events
        self._max_events = max_events
        self._num_velocity_bins = num_velocity_bins
        self._note_performance = note_performance

    def transform(self, input_object):
        quantized_sequence = input_object
        performances = extract_performances(
            quantized_sequence,
            min_events_discard=self._min_events,
            max_events_truncate=self._max_events,
            num_velocity_bins=self._num_velocity_bins,
            note_performance=self._note_performance)
        return performances
  
def extract_performances(
    quantized_sequence, start_step=0, min_events_discard=None,
    max_events_truncate=None, max_steps_truncate=None, num_velocity_bins=0,
    split_instruments=False, note_performance=False):
    """Extracts one or more performances from the given quantized NoteSequence.

    Args:
    quantized_sequence: A quantized NoteSequence.
    start_step: Start extracting a sequence at this time step.
    min_events_discard: Minimum length of tracks in events. Shorter tracks are
        discarded.
    max_events_truncate: Maximum length of tracks in events. Longer tracks are
        truncated.
    max_steps_truncate: Maximum length of tracks in quantized time steps. Longer
        tracks are truncated.
    num_velocity_bins: Number of velocity bins to use. If 0, velocity events
        will not be included at all.
    split_instruments: If True, will extract a performance for each instrument.
        Otherwise, will extract a single performance.
    note_performance: If True, will create a NotePerformance object. If
        False, will create either a MetricPerformance or Performance based on
        how the sequence was quantized.

    Returns:
    performances: A python list of Performance or MetricPerformance (if
        `quantized_sequence` is quantized relative to meter) instances.
    stats: A dictionary mapping string names to `statistics.Statistic` objects.
    """
    note_seq.sequences_lib.assert_is_quantized_sequence(quantized_sequence)


    if note_seq.sequences_lib.is_absolute_quantized_sequence(quantized_sequence):
        steps_per_second = quantized_sequence.quantization_info.steps_per_second
        # Create a histogram measuring lengths in seconds.
        #stats['performance_lengths_in_seconds'] = note_seq.statistics.Histogram(
        #    'performance_lengths_in_seconds',
        #    [5, 10, 20, 30, 40, 60, 120])
    else:
        steps_per_bar = note_seq.sequences_lib.steps_per_bar_in_quantized_sequence(
            quantized_sequence)
        # Create a histogram measuring lengths in bars.
        #stats['performance_lengths_in_bars'] = note_seq.statistics.Histogram(
        #    'performance_lengths_in_bars',
        #    [1, 10, 20, 30, 40, 50, 100, 200, 500])

    if split_instruments:
        instruments = set(note.instrument for note in quantized_sequence.notes)
    else:
        instruments = set([None])
        # Allow only 1 program.
        programs = set()
        for note in quantized_sequence.notes:
            programs.add(note.program)
        if len(programs) > 1:
            #stats['performances_discarded_more_than_1_program'].increment()
            return []

    performances = []

    for instrument in instruments:
        # Translate the quantized sequence into a Performance.
        if note_performance:
            try:
                performance = note_seq.NotePerformance(
                    quantized_sequence, start_step=start_step,
                    num_velocity_bins=num_velocity_bins, instrument=instrument)
            except note_seq.TooManyTimeShiftStepsError:
                #stats['performance_discarded_too_many_time_shift_steps'].increment()
                continue
            except note_seq.TooManyDurationStepsError:
                #stats['performance_discarded_too_many_duration_steps'].increment()
                continue
        elif note_seq.sequences_lib.is_absolute_quantized_sequence(quantized_sequence):
            performance = note_seq.Performance(quantized_sequence, start_step=start_step,
                                    num_velocity_bins=num_velocity_bins,
                                    instrument=instrument)
        else:
            performance = note_seq.MetricPerformance(quantized_sequence, start_step=start_step,
                                            num_velocity_bins=num_velocity_bins,
                                            instrument=instrument)

        if (max_steps_truncate is not None and
            performance.num_steps > max_steps_truncate):
            performance.set_length(max_steps_truncate)
            #stats['performances_truncated_timewise'].increment()

        if (max_events_truncate is not None and
            len(performance) > max_events_truncate):
            performance.truncate(max_events_truncate)
            #stats['performances_truncated'].increment()

        if min_events_discard is not None and len(performance) < min_events_discard:
            #stats['performances_discarded_too_short'].increment()
            pass
        else:
            performances.append(performance)
            if note_seq.sequences_lib.is_absolute_quantized_sequence(quantized_sequence):
                #stats['performance_lengths_in_seconds'].increment(
                #    performance.num_steps // steps_per_second)
                pass
            else:
               pass
                #stats['performance_lengths_in_bars'].increment(
                #    performance.num_steps // steps_per_bar)

    return performances

class NoteSequencePipeline():
  """Superclass for pipelines that input and output NoteSequences."""

  def __init__(self, name=None):
    """Construct a NoteSequencePipeline. Should only be called by subclasses.

    Args:
      name: Pipeline name.
    """
    #super(NoteSequencePipeline, self).__init__(
    #    input_type=music_pb2.NoteSequence,
    #    output_type=music_pb2.NoteSequence,
    #    name=name)
    pass

class Quantizer(NoteSequencePipeline):
  """A Pipeline that quantizes NoteSequence data."""

  def __init__(self, steps_per_quarter=None, steps_per_second=None, name=None):
    """Creates a Quantizer pipeline.

    Exactly one of `steps_per_quarter` and `steps_per_second` should be defined.

    Args:
      steps_per_quarter: Steps per quarter note to use for quantization.
      steps_per_second: Steps per second to use for quantization.
      name: Pipeline name.

    Raises:
      ValueError: If both or neither of `steps_per_quarter` and
          `steps_per_second` are set.
    """
    super(Quantizer, self).__init__(name=name)
    if (steps_per_quarter is not None) == (steps_per_second is not None):
      raise ValueError(
          'Exactly one of steps_per_quarter or steps_per_second must be set.')
    self._steps_per_quarter = steps_per_quarter
    self._steps_per_second = steps_per_second

  def transform(self, input_object):
    note_sequence = input_object
    try:
      if self._steps_per_quarter is not None:
        quantized_sequence = note_seq.sequences_lib.quantize_note_sequence(
            note_sequence, self._steps_per_quarter)
      else:
        quantized_sequence = note_seq.sequences_lib.quantize_note_sequence_absolute(
            note_sequence, self._steps_per_second)
      return [quantized_sequence]
    except note_seq.sequences_lib.MultipleTimeSignatureError as e:
      #tf.logging.warning('Multiple time signatures in NoteSequence %s: %s',
      #                   note_sequence.filename, e)
      #self._set_stats([note_seq.statistics.Counter(
      #    'sequences_discarded_because_multiple_time_signatures', 1)])
      return []
    except note_seq.sequences_lib.MultipleTempoError as e:
      #tf.logging.warning('Multiple tempos found in NoteSequence %s: %s',
      #                   note_sequence.filename, e)
      #self._set_stats([note_seq.statistics.Counter(
      #    'sequences_discarded_because_multiple_tempos', 1)])
      return []
    except note_seq.sequences_lib.BadTimeSignatureError as e:
      #tf.logging.warning('Bad time signature in NoteSequence %s: %s',
      #                   note_sequence.filename, e)
      #self._set_stats([note_seq.statistics.Counter(
      #    'sequences_discarded_because_bad_time_signature', 1)])
      return []
    
class TranspositionPipeline(NoteSequencePipeline):
  """Creates transposed versions of the input NoteSequence."""

  def __init__(self, transposition_range, min_pitch=note_seq.constants.MIN_MIDI_PITCH,
               max_pitch=note_seq.constants.MAX_MIDI_PITCH, name=None):
    """Creates a TranspositionPipeline.

    Args:
      transposition_range: Collection of integer pitch steps to transpose.
      min_pitch: Integer pitch value below which notes will be considered
          invalid.
      max_pitch: Integer pitch value above which notes will be considered
          invalid.
      name: Pipeline name.
    """
    super(TranspositionPipeline, self).__init__(name=name)
    self._transposition_range = transposition_range
    self._min_pitch = min_pitch
    self._max_pitch = max_pitch

  def transform(self, input_object):
    sequence = input_object

    transposed = []
    for amount in self._transposition_range:
      # Note that transpose is called even with a transpose amount of zero, to
      # ensure that out-of-range pitches are handled correctly.
      ts = self._transpose(sequence, amount)
      if ts is not None:
        transposed.append(ts)

    return transposed

  def _transpose(self, ns, amount):
    """Transposes a note sequence by the specified amount."""
    ts = copy.deepcopy(ns)
    for note in ts.notes:
      if not note.is_drum:
        note.pitch += amount
        if note.pitch < self._min_pitch or note.pitch > self._max_pitch:
          # skipped_due_to_range_exceeded
          return None
    return ts