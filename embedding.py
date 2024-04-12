import note_seq, pretty_midi
from note_seq import midi_io, sequences_lib
import numpy as np
from data_loader import load_data
from pipelines import PerformanceExtractor, extract_performances, Quantizer, TranspositionPipeline
import pickle

def load_embedding_from_pickle(args):
    try:
        data_embedding = pickle.load(open(f'data/{args.data}_embedding.p', "rb"))
        args.log("Data Embedding loaded from pickle file")
    except:
        args.log("No embedding pickle file found, loading raw data and embedding it")
        try:
            data = pickle.load(open(f'data/{args.data}.p', "rb"))
            args.log("Raw Data loaded from pickle file")
        except:
            args.log("No raw data pickle file found, loading raw data from midi files")
            data_path = 'data/maestro-v3.0.0'
            data = load_data(args, data_path)
            args.log("Raw Data loaded")
            pickle.dump(data, open(f'data/{args.data}.p', "wb"))
            args.log("Raw Data saved to pickle file")

        args.log("Begining Embedding Data")
        data_embedding = []
        for midi_seq in data:
            data_embedding.append(embedding(midi_seq))
        args.log("Data Embedding Complete")
        pickle.dump(data_embedding, open(f'data/{args.data}_embedding.p', "wb"))
        args.log("Data Embedding saved to pickle file")

    return data_embedding
'''
def embed(note_seqs, mode='train'):
    # note_seqs, list of NoteSequence
    stretch_factors = [0.95, 0.975, 1.0, 1.025, 1.05] if mode == 'train' else [1.0]
    hop_size_seconds = 30.0
    # Traponse no more than a major third
    transposition_range = list(range(-3, 4)) if mode == 'training' else [0]

    for note_seq in note_seqs:
        
        # Apply sustain control changes
        note_seq = sequences_lib.apply_sustain_control_changes(note_seq)

        # Apply note stretches up to 5% either direction in time
        note_seq = [sequences_lib.stretch_note_sequence(note_seq, stretch_factor)
            for stretch_factor in stretch_factors]
        
        # Split into chunks of 30 seconds
        note_seq = [
            sequences_lib.split_note_sequence(n, hop_size_seconds)
            for n in note_seq
            ]

        # Flatten into many sequences
        note_seq = [n for ns in note_seq for n in ns]
        print(len(note_seq), type(note_seq[0]))

        # Quantize to 100 steps per second
        quantizer = Quantizer(steps_per_second=100, name='Quantizer')
        note_seq = [quantizer.transform(n) for n in note_seq]
        note_seq = [n for ns in note_seq for n in ns] # Flatten

        print(len(note_seq), type(note_seq[0]))

        # Transpose up to a major third in either direction
        transposition_pipeline  = TranspositionPipeline(transposition_range)
        note_seq = [transposition_pipeline.transform(n) for n in note_seq]
        note_seq = [n for ns in note_seq for n in ns] # Flatten

        print(len(note_seq), type(note_seq[0]))

        perf_extractor = PerformanceExtractor(
        min_events=32,
        max_events=512,
        num_velocity_bins=0,
        note_performance=False,
        )
        note_seq = [perf_extractor.transform(n) for n in note_seq]
        note_seq = [n for ns in note_seq for n in ns] # Flatten

        print(len(note_seq), type(note_seq[0]))


        #encoder_decoder = note_seq.OneHotEventSequenceEncoderDecoder(note_seq.PerformanceOneHotEncoding())
        #processed = encoder_decoder.encode(x)
        print(processed)
        
        
    #encoder_decoder = note_seq.OneHotEventSequenceEncoderDecoder(note_seq.PerformanceOneHotEncoding())
    #encoder_decoder.encode()
'''
def embedding(midi_seq):
    # Convert notes into an embedding for a transformer model:
    notes_array = np.array(midi_seq.notes)

    # Determine the maximum pitch and velocity for normalization

    input_vector = np.zeros((len(notes_array), 5))

    input_vector[0][0] = (notes_array[0].end_time - notes_array[0].start_time)
    input_vector[0][1] = 0  # Time between notes
    input_vector[0][2] = 0
    input_vector[0][3] = notes_array[0].pitch
    input_vector[0][4] = notes_array[0].velocity

    for i in range(1, len(notes_array)):
        input_vector[i][0] = (notes_array[i].end_time - notes_array[i].start_time) 
        input_vector[i][1] = (notes_array[i].start_time - notes_array[i-1].end_time)
        input_vector[i][2] = (notes_array[i].start_time - notes_array[i-1].start_time)
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
    end_time_previous_note = 0
    for i in range(len(embedding)):
        start_time = 0.5 * (start_time_previous_note + embedding[i][2]) + 0.5 * (end_time_previous_note + embedding[i][1])
        end_time = start_time + embedding[i][0]
        note = pretty_midi.Note(velocity=int(embedding[i][4]), pitch=int(embedding[i][3]), start=float(start_time), end=float(end_time))
        notes_array.append(note)
        start_time_previous_note = start_time
        end_time_previous_note = end_time
    return notes_array


def decode_embedding_discritized(embedding):
    notes_array = []
    start_time_previous_note = 0
    # print("Embedding shape:", embedding.shape)  # Verify the shape of embedding

    for i in range(embedding.shape[1]):  # embedding.shape[1] should be the number of notes if the first dimension is batch
        # print("Current embedding element shape:", embedding[0, i].shape)  # Check the shape of each note's features

        start_increment = embedding[0, i, 2].item()
        start_time = start_time_previous_note + start_increment
        end_time = start_time + embedding[0, i, 0].item()

        # Convert pitch to int explicitly
        # print('embedding[0, i, 3].item()', embedding[0, i, 3].item())
        pitch = int(embedding[0, i, 3].item())
        # print('pitch', pitch)
        # Convert velocity to int explicitly
        velocity = int(embedding[0, i, 4].item())

        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,  # Use the integer pitch
            start=float(start_time),
            end=float(end_time)
        )
        notes_array.append(note)
        start_time_previous_note = start_time

    return notes_array

if __name__ == '__main__':
    from data_loader import load_file
    data_path = 'data/maestro-v3.0.0'
    test_path = '/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi'
    

    midi_seq = load_file(data_path + test_path)
    #print(midi_seq)

    #output = embed([midi_seq])

    
    ##print(midi_seq)
    input_vector = embedding(midi_seq)

    # Only 3 Decimals:
    ##np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    ##print(input_vector)

    embedding_to_midi(input_vector, 'output.mid')
