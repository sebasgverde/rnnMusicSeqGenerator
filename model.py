# --------------------------------------------
# Copyright (c) 2018 sebastian garcia valencia
# --------------------------------------------

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
from tensorflow.contrib.tensorboard.plugins import projector

import os
import numpy as np


class Model():
    def __init__(self, args, training=True):
        self.args = args
        if not training:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        elif args.model == 'nas':
            cell_fn = rnn.NASCell
        elif args.model == 'ugrnn':
            cell_fn = rnn.UGRNNCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cells = []

        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size)
            if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
                cell = rnn.DropoutWrapper(cell,
                                          input_keep_prob=args.input_keep_prob,
                                          output_keep_prob=args.output_keep_prob)
            cells.append(cell)

        # you can also manually create combinations of different cell types

        # for _ in range(2):
        #     cell = cell_fn(args.rnn_size)
        #     if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
        #         cell = rnn.DropoutWrapper(cell,
        #                                   input_keep_prob=args.input_keep_prob,
        #                                   output_keep_prob=args.output_keep_prob)
        #     cells.append(cell)

        # # this works because state is also a lstmstatetuple
        # for _ in range(2):
        #     cell = rnn.NASCell(args.rnn_size)
        #     if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
        #         #this works for any RNN cell (tf.contrib.rnn)
        #         cell = rnn.DropoutWrapper(cell,
        #                                   input_keep_prob=args.input_keep_prob,
        #                                   output_keep_prob=args.output_keep_prob)
        #     cells.append(cell)
        # # in this case state is only an array, so should be done different
        # for _ in range(2):
        #     cell = rnn.GRUCell(args.rnn_size)
        #     if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
        #         #this works for any RNN cell (tf.contrib.rnn)
        #         cell = rnn.DropoutWrapper(cell,
        #                                   input_keep_prob=args.input_keep_prob,
        #                                   output_keep_prob=args.output_keep_prob)
        #     cells.append(cell)

        # for _ in range(2):
        #     cell = rnn.BasicLSTMCell(args.rnn_size)
        #     if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
        #         #this works for any RNN cell (tf.contrib.rnn)
        #         cell = rnn.DropoutWrapper(cell,
        #                                   input_keep_prob=args.input_keep_prob,
        #                                   output_keep_prob=args.output_keep_prob)
        #     cells.append(cell)        

        # self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

        # for _ in range(2):
        #     cell = rnn.UGRNNCell(args.rnn_size)
        #     if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
        #         #this works for any RNN cell (tf.contrib.rnn)
        #         cell = rnn.DropoutWrapper(cell,
        #                                   input_keep_prob=args.input_keep_prob,
        #                                   output_keep_prob=args.output_keep_prob)
        #     cells.append(cell)        

        # for _ in range(1):
        #     cell = rnn.NASCell(args.rnn_size)
        #     if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
        #         #this works for any RNN cell (tf.contrib.rnn)
        #         cell = rnn.DropoutWrapper(cell,
        #                                   input_keep_prob=args.input_keep_prob,
        #                                   output_keep_prob=args.output_keep_prob)
        #     cells.append(cell)    
            
        # for _ in range(1):
        #     cell = rnn.GRUCell(args.rnn_size)
        #     if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
        #         #this works for any RNN cell (tf.contrib.rnn)
        #         cell = rnn.DropoutWrapper(cell,
        #                                   input_keep_prob=args.input_keep_prob,
        #                                   output_keep_prob=args.output_keep_prob)
        #     cells.append(cell)        

        self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

        self.input_data = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])
        # The state of each cell at the final time-step. It is a 2D Tensor of shape [batch_size x cell.state_size]. (Note that in some cases, 
        # like basic RNN cell or GRU cell, outputs and states can be the same. They are different for LSTM cells though.)
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w",
                                        [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])



        #--------------------------------------------------------------------
        #this segments is the key for the representation changes
        embedding_var = tf.get_variable("embedding_var", [args.vocab_size, args.rnn_size])
        # use input data as indexes and return the corresponding elements in embedding, it is thougt for words 
        # (https://www.tensorflow.org/tutorials/word2vec), so the only way i see to use it is to see tuples (note, time)
        # as a word with unique index, and any harmony (maybe a special tuple for simultaneos time change)
        inputs = tf.nn.embedding_lookup(embedding_var, self.input_data)

        # dropout beta testing: double check which one should affect next line
        if training and args.output_keep_prob:
            inputs = tf.nn.dropout(inputs, args.output_keep_prob)

        inputs = tf.split(inputs, args.seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
        #--------------------------------------------------------------------

        if training:

            #embedding porjector
            # Format: tensorflow/tensorboard/plugins/projector/projector_config.proto
            config = projector.ProjectorConfig()

            # You can add multiple embeddings. Here we add only one.
            embedding = config.embeddings.add()
            embedding.tensor_name = "embedding_var"
            # Link this tensor to its metadata file (e.g. labels).
            embedding.metadata_path = os.path.join(args.log_dir, 'metadata.tsv')

            # Use the same LOG_DIR where you stored your checkpoint.
            summary_writer = tf.summary.FileWriter(args.complete_log_dir)

            # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
            # read this file during startup.
            projector.visualize_embeddings(summary_writer, config)



        # loop_function: If not None, this function will be applied to the i-th output in order to generate the i+1-st input, 
        # and decoder_inputs will be ignored, except for the first element ("GO" symbol). 
        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding_var, prev_symbol)

        # outputs, states = basic_rnn_seq2seq(encoder_inputs, decoder_inputs, cell)
        # In the above call, encoder_inputs are a list of tensors representing inputs to the encoder, i.e., corresponding to the 
        # letters A, B, C in the first picture above. Similarly, decoder_inputs are tensors representing inputs to the decoder, 
        # GO, W, X, Y, Z on the first picture.

        # The cell argument is an instance of the tf.contrib.rnn.RNNCell class that determines which cell will be used inside the model. 
        # You can use an existing cell, such as GRUCell or LSTMCell, or you can write your own. Moreover, tf.contrib.rnn provides wrappers 
        # to construct multi-layer cells, add dropout to cell inputs or outputs, or to do other transformations. See the RNN Tutorial for 
        # examples.
        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if not training else None, scope='rnnlm')
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])


        # this part is just a classic linear regression and optimization over the result of the decoder above
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        loss = legacy_seq2seq.sequence_loss_by_example(
                [self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])])
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        with tf.name_scope('cost'):
            self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # instrument tensorboard
        tf.summary.histogram('logits', self.logits)
        tf.summary.histogram('loss', loss)
        tf.summary.scalar('train_loss', self.cost)


    # weighted rulette method to select the sample
    def roulette_wheel_selection(self, weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        return(int(np.searchsorted(t, np.random.rand(1)*s)))

    # 60, 62 is central C4, D4
    def sample_midi(self, sess, notes_dict, inverse_notes_dict, num, seed, sampling_mode, selection_method):

            # update of the state with the seed, moving the weigths in gates but
            # without make inference for the first n-1
            if sampling_mode == 'melody':
                seed = [int(note) for note in seed.split(' ')]
            elif sampling_mode == 'words':
                seed = seed.split(' ')
            elif sampling_mode == 'melody_interval':
                # [1:] cause the first element in seed is the base note, the 
                # others are the intervals
                seed = [int(note) for note in seed.split(' ')[1:]]

            if selection_method == 'max':
                selection_fn = np.argmax
            else:
                selection_fn = self.roulette_wheel_selection 
                

            state = sess.run(self.cell.zero_state(1, tf.float32))
            for note in seed[:-1]:
                x = np.zeros((1, 1))
                x[0, 0] = inverse_notes_dict[note]
                feed = {self.input_data: x, self.initial_state: state}
                [state] = sess.run([self.final_state], feed)


            ret = seed

            #this should be the emdding position, as represented in notes_dict
            note = inverse_notes_dict[seed[-1]]
            for n in range(num):
                x = np.zeros((1, 1))
                x[0, 0] = note
                feed = {self.input_data: x, self.initial_state: state}
                [probs, state] = sess.run([self.probs, self.final_state], feed)
                p = probs[0]

                sample = selection_fn(p)

                pred = sample
                ret.append(notes_dict[pred])
                note = pred
            return ret
