# --------------------------------------------
# Copyright (c) 2018 sebastian garcia valencia
# --------------------------------------------

from __future__ import print_function
import tensorflow as tf

import argparse
import os
import cPickle

from model import Model

from midi_manager import MidiWriter


def main():
    parser = argparse.ArgumentParser(
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ckpt_dir', type=str, default='save',
                        help='model directory where are checkpointed models')
    parser.add_argument('-n', type=int, default=500,
                        help='number of sequence elements to generate')
    parser.add_argument('--seed', type=str, default='60 62',
                        help='seed midi notes')
    parser.add_argument('--selection_method', type=str, default='roulette',
                        help='roulette or max')
    parser.add_argument('--output_uri', type=str, default='output.mid',
                        help='uri to the output midi file')
    parser.add_argument('--sampling_mode', type=str, default='melody',
                        help='mode of sampling, melody, harmony or time')
    parser.add_argument('--ckpt_file', type=str, default='None',
                        help='a ckpt complete uri, use it if you want to use a specific '
                             'ckpt instead of the last in checkpoint file')     

    args = parser.parse_args()

    sample_midi(args)

def sample_midi(args):
    midi_writer = MidiWriter()
    with open(os.path.join(args.ckpt_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.ckpt_dir, 'notes_dict.pkl'), 'rb') as f:
        notes_dict = cPickle.load(f)     

    inverse_notes_dict = dict(zip(notes_dict.values(), notes_dict))
    model = Model(saved_args, training=False)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            if args.ckpt_file != 'None':
                saver.restore(sess, args.ckpt_file)
            else:
                saver.restore(sess, ckpt.model_checkpoint_path)

            sequence = model.sample_midi(sess, notes_dict, inverse_notes_dict, args.n, args.seed,
                               args.sampling_mode, args.selection_method)

            print(sequence)

            print('sampling mode: ' + args.sampling_mode)

            if args.sampling_mode == 'melody':
                midi_writer.sequenceVector2midiMelody(sequence, args.output_uri)
            elif args.sampling_mode == 'time':
                midi_writer.sequenceVector2midiMelodyWithTime(sequence, args.output_uri)                
            elif args.sampling_mode == 'harmony':
                midi_writer.sequenceVector2midiHarmony(sequence, args.output_uri)
            elif args.sampling_mode == 'words':
                midi_writer.sequenceVector2wordsText(sequence, args.output_uri)
            # midi_writer.sequenceVector2midiWithTime(sequence, args.output_uri)
            elif args.sampling_mode == 'melody_interval':
             
                midi_writer.sequenceVectorInterval2midiMelody(sequence, args.output_uri, int(args.seed.split(' ')[0]))

if __name__ == '__main__':
    main()
