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
        midi_files = [f for f in files if f.endswith('.midi')]
        t = tqdm(midi_files, desc="Loading MIDI files", leave=True)
        for file in t:
            file_path = os.path.join(subdir, file)
            i += 1
            data.append(load_file(file_path))
            t.set_description(f"Processing {file} ({i}/{len(midi_files)})")
            if args.tiny and i > 4: break
    return np.array(data)

def load_file(file_path):
    #print("LOADING FILE: " + file_path)
    midi_data = pretty_midi.PrettyMIDI(file_path)
    midi_seq = midi_io.midi_to_note_sequence(midi_data)
    return midi_seq

if __name__ == '__main__':
    data = load_data()
    print(data)