import note_seq, pretty_midi
from note_seq import midi_io
import numpy as np

if __name__ == '__main__':
    data_path = 'data/maestro-v3.0.0'
    test_path = '/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi'
    midi_data = pretty_midi.PrettyMIDI(data_path + test_path)
    midi_seq = midi_io.midi_to_note_sequence(midi_data)

    ##print(midi_seq.notes)

    # Converet notes into a an embedding that can be interpreted by a transformer model:

    notes_array = np.array(midi_seq.notes)

    input_vector = np.zeros((len(notes_array), 5))

    input_vector[0][0] = notes_array[0].end_time - notes_array[0].start_time
    input_vector[0][1] = 0 # Time between notes
    input_vector[0][2] = 0
    input_vector[0][3] = notes_array[0].pitch
    input_vector[0][4] = notes_array[0].velocity

    for i in range(1, len(notes_array)):
        ##input_vector[i][0] = notes_array[i].start_time
        ##input_vector[i][1] = notes_array[i].end_time
        input_vector[i][0] = notes_array[i].end_time - notes_array[i].start_time
        input_vector[i][1] = notes_array[i].start_time - notes_array[i-1].end_time # Time between notes
        input_vector[i][2] = notes_array[i].start_time - notes_array[i-1].start_time # Time since last note started
        input_vector[i][3] = notes_array[i].pitch
        input_vector[i][4] = notes_array[i].velocity


    # Only 3 Decimals:
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    print(input_vector)