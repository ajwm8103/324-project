import note_seq, pretty_midi
from note_seq import midi_io

if __name__ == '__main__':
    data_path = 'data/maestro-v3.0.0'
    test_path = '/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi'
    midi_data = pretty_midi.PrettyMIDI(data_path + test_path)
    midi_seq = midi_io.midi_to_note_sequence(midi_data)

    print(midi_seq)
