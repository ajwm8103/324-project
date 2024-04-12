import note_seq, pretty_midi
from note_seq import midi_io
import os
import numpy as np
from tqdm import tqdm

def load_data(args, data_path='data/maestro-v3.0.0'):
    # Loop through every file in the folder and load each file
    data = []
    i = 0
    for subdir, dirs, files in os.walk(data_path):
        for file in tqdm(files):
            file_path = os.path.join(subdir, file)
            if file_path.endswith(".midi"):
                i += 1
                data.append(load_file(args, file_path))
            if args.tiny and i > 30: break
    return np.array(data)

def load_file(args, file_path):
    args.log("LOADING FILE: " + file_path)
    midi_data = pretty_midi.PrettyMIDI(file_path)
    midi_seq = midi_io.midi_to_note_sequence(midi_data)
    return midi_seq

if __name__ == '__main__':
    data = load_data()
    print(data)