import argparse

import note_seq, pretty_midi
from note_seq import midi_io

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-v", "--verbose", action="store_true",
                    help="increase output verbosity")
                    

    args = parser.parse_args()
    
    print('hi')

    if args.verbose:
        print("verbose")
    else:
        print("not verbose")
