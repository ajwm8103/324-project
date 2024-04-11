import note_seq, pretty_midi
from note_seq import midi_io
import numpy as np
from data_loader import load_data
import pickle

def load_embedding_from_pickle(verbose=False):
    try:
        data_embedding = pickle.load(open("data_embedding.p", "rb"))
        if verbose: print("Data Embedding loaded from pickle file")    
    except:
        if verbose: print("No embedding pickle file found, loading raw data and embedding it")
        try:
            data = pickle.load(open("data.p", "rb"))
            if verbose: print("Raw Data loaded from pickle file")
        except:
            if verbose: print("No raw data pickle file found, loading raw data from midi files")
            data_path = 'data/maestro-v3.0.0'
            data = load_data(data_path)
            if verbose: print("Raw Data loaded")
            pickle.dump(data, open("data.p", "wb"))
            if verbose: print("Raw Data saved to pickle file")

        if verbose: print("Begining Embedding Data")
        data_embedding = []
        for midi_seq in data:
            data_embedding.append(embedding(midi_seq))
        if verbose: print("Data Embedding Complete")
        pickle.dump(data_embedding, open("data_embedding.p", "wb"))
        if verbose: print("Data Embedding saved to pickle file")

    return data_embedding

def embedding(midi_seq):
    # Converet notes into a an embedding that can be interpreted by a transformer model:
    notes_array = np.array(midi_seq.notes)

    input_vector = np.zeros((len(notes_array), 5))

    input_vector[0][0] = notes_array[0].end_time - notes_array[0].start_time
    input_vector[0][1] = 0 # Time between notes
    input_vector[0][2] = 0
    input_vector[0][3] = notes_array[0].pitch
    input_vector[0][4] = notes_array[0].velocity

    for i in range(1, len(notes_array)):
        input_vector[i][0] = notes_array[i].end_time - notes_array[i].start_time
        input_vector[i][1] = notes_array[i].start_time - notes_array[i-1].end_time # Time between notes
        input_vector[i][2] = notes_array[i].start_time - notes_array[i-1].start_time # Time since last note started
        input_vector[i][3] = notes_array[i].pitch
        input_vector[i][4] = notes_array[i].velocity

    return input_vector



if __name__ == '__main__':
    data_path = 'data/maestro-v3.0.0'
    test_path = '/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi'
    input_vector = embedding(pretty_midi.PrettyMIDI(data_path + test_path))


    # Only 3 Decimals:
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    print(input_vector)