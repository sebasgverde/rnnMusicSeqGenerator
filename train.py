# --------------------------------------------
# Copyright (c) 2018 sebastian garcia valencia
# --------------------------------------------

from __future__ import print_function
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

from model import Model

from data_processor import DataProcessor


def main():
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data',
                        help='directory containing data')
    parser.add_argument('--data_style', type=str, default='pickle',
                        help='dataset style: pickle, folder, piano_roll')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='directory to store tensorboard logs')
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, lstm, nas or ugrnn')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=50,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='save frequency')
    parser.add_argument('--max_models_keep', type=int, default=5,
                        help='max number of models conserved')    
    parser.add_argument('--grad_clip', type=float, default=5.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                        help='decay rate for rmsprop')
    parser.add_argument('--output_keep_prob', type=float, default=1.0,
                        help='probability of keeping weights in the hidden layer')
    parser.add_argument('--input_keep_prob', type=float, default=1.0,
                        help='probability of keeping weights in the input layer')
    parser.add_argument('--training_mode', type=str, default='melody',
                        help='mode of training, melody, harmony or time')    
    parser.add_argument('--init_from', type=str, default=None,
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'chars_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    args = parser.parse_args()
    train(args)


def train(args):
    data_proc = DataProcessor(args.data_dir, args.batch_size, args.seq_length, args.training_mode, args.data_style)
    args.vocab_size = data_proc.vocab_size

    args.complete_log_dir = os.path.join(args.log_dir, time.strftime("%Y-%m-%d-%H-%M-%S"))

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'notes_dict.pkl'), 'wb') as f:
        cPickle.dump((data_proc.notes_dict), f)

    with open(os.path.join(args.log_dir, 'metadata.tsv'), 'wb') as f:
        f.write((data_proc.dict2metadataTsvFile()))


    model = Model(args)

    with tf.Session() as sess:
        # instrument for tensorboard
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(
                args.complete_log_dir)
        writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=args.max_models_keep)
        # restore model
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for e in range(args.num_epochs):
            sess.run(tf.assign(model.lr,
                               args.learning_rate * (args.decay_rate ** e)))

            data_proc.reset_batch_pointer()

            state = sess.run(model.initial_state)

            for b in range(data_proc.num_batches):
                start = time.time()

                x, y = data_proc.next_batch()
                feed = {model.input_data: x, model.targets: y}

                # updates the weights of the RNNs to use it as feed in the next step
                # I modified it to be able to use more cells types, some like basic rnn or GRU doesn't have a state tuple
                # but a single array and it was giving me an error, with this try catch, if there is no c and h, which are
                # the components of the lstmstatetuple (used by lstm, nas and others) it takes the single array, it works
                # because all is divided in state, so the i make sense and it works diffrent internally
                for i, state_tuple in enumerate(model.initial_state):
                    # I had the doubt why it doesn't overwrite feed[c] and feed[h], is because c and h are not
                    # the actual arrays they are the class name (something like 'MultiRNNCellZeroState/GRUCellZeroState/zeros:0')
                    # which is used as key and then the actual array is taken form state, the magical python dics :)
                    try:
                        feed[state_tuple.c] = state[i].c
                        feed[state_tuple.h] = state[i].h
                    except:
                        feed[state_tuple] = state[i]

                train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)


                # instrument for tensorboard
                summ, train_loss, state, _ = sess.run([summaries, model.cost, model.final_state, model.train_op], feed)
                writer.add_summary(summ, e * data_proc.num_batches + b)

                end = time.time()
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                      .format(e * data_proc.num_batches + b,
                              args.num_epochs * data_proc.num_batches,
                              e, train_loss, end - start))
                if (e * data_proc.num_batches + b) % args.save_every == 0\
                        or (e == args.num_epochs-1 and
                            b == data_proc.num_batches-1):
                    # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt_loss_' + str(train_loss))
                    saver.save(sess, checkpoint_path,
                               global_step=e * data_proc.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))


if __name__ == '__main__':
    main()
