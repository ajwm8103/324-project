import note_seq, pretty_midi
from note_seq import midi_io
import numpy as np
from data_loader import load_data
import pickle

def load_embedding_from_pickle(args):
    try:
        data_embedding = pickle.load(open("data/data_embedding.p", "rb"))
        args.log("Data Embedding loaded from pickle file")    
    except:
        args.log("No embedding pickle file found, loading raw data and embedding it")
        try:
            data = pickle.load(open("data/data.p", "rb"))
            args.log("Raw Data loaded from pickle file")
        except:
            args.log("No raw data pickle file found, loading raw data from midi files")
            data_path = 'data/maestro-v3.0.0'
            data = load_data(args, data_path)
            args.log("Raw Data loaded")
            pickle.dump(data, open("data/data.p", "wb"))
            args.log("Raw Data saved to pickle file")

        args.log("Begining Embedding Data")
        data_embedding = []
        for midi_seq in data:
            data_embedding.append(embedding(midi_seq))
        args.log("Data Embedding Complete")
        pickle.dump(data_embedding, open("data/data_embedding.p", "wb"))
        args.log("Data Embedding saved to pickle file")

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


def embedding_to_midi(embedding, filename='output.mid'):
    notes = decode_embedding(embedding)
    for note in notes:
        assert isinstance(note.pitch, int), f"Pitch must be int, got {type(note.pitch)}"
        assert isinstance(note.velocity, int), f"Velocity must be int, got {type(note.velocity)}"
        assert isinstance(note.start, (float, int)), f"Start must be float or int, got {type(note.start)}"
        assert isinstance(note.end, (float, int)), f"End must be float or int, got {type(note.end)}"

    midi_file = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0) # Default the instrument to Piano
    instrument.notes = notes # TODO: extend?
    midi_file.instruments.append(instrument)
    midi_file.write(filename)

def decode_embedding(embedding):
    notes_array = []
    start_time_previous_note = 0
    for i in range(len(embedding)):
        start_time = start_time_previous_note + embedding[i][2] # TODO: This isn't perfect due to redundancy, maybe do an average?
        end_time = start_time + embedding[i][0]
        note = pretty_midi.Note(velocity=int(embedding[i][4]), pitch=int(embedding[i][3]), start=float(start_time), end=float(end_time))
        notes_array.append(note)
        start_time_previous_note = start_time
    return notes_array

if __name__ == '__main__':
    from data_loader import load_file
    data_path = 'data/maestro-v3.0.0'
    test_path = '/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi'
    
    midi_seq = load_file(data_path + test_path)
    ##print(midi_seq)
    input_vector = embedding(midi_seq)

    # Only 3 Decimals:
    ##np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    ##print(input_vector)

    embedding_to_midi(input_vector, 'output.midi')
