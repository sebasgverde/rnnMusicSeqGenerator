# --------------------------------------------
# Copyright (c) 2018 sebastian garcia valencia
# --------------------------------------------

import midi
import numpy as np
import os
from midiutil.MidiFile import MIDIFile
import cPickle
import pickle
import math
from midi_manager import MidiLoader

class DataProcessor():
    def __init__(self, data_dir, batch_size, seq_length, training_mode, data_style):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.visualization_type = 0
        self.midi_loader = MidiLoader(data_dir)

        print 'training mode: ' + training_mode 

        if training_mode != 'test':
            tensor = []
            if training_mode == 'melody':
                # tensor = self.read_piano_roll_dataset_as_melody()
                # tensor = self.read_dataset_as_melody()
                if data_style == 'folder':
                    tensor = self.midi_loader.midi_folder_2_list_of_sequences()
                self.visualization_type = 0
            elif training_mode =='harmony':
                tensor = self.midi_loader.read_piano_roll_dataset_as_harmony()
                self.visualization_type = 1
            elif training_mode =='time':
                tensor = self.midi_loader.read_piano_roll_dataset_as_melody_with_time()
                self.visualization_type = 2
            elif training_mode == 'words':
                tensor = self.midi_loader.text_words_folder_2_list_of_sequences()
                self.visualization_type = 3
            elif training_mode == 'melody_interval':
                self.visualization_type = 3

            # with the changes to use pickles as dataset the support for time, harmony is not
            #working, I have to fix it
            # tensor = self.midi_loader.read_dataset_as_melody_with_time()
            
            if data_style == 'folder':
                self.create_dict_and_training_data(tensor)
            else:
                self.create_dict_and_training_data_from_pickle()
            
            # self.create_dict_and_training_data_from_pickle()
            self.create_batches()
            self.reset_batch_pointer()

    def dict2metadataTsvFile(self):
        general_string = ''

        if self.visualization_type == 0:
            for key in self.notes_dict:
                general_string += str(self.midi_loader.inverse_midi_note_dict[self.notes_dict[key]]) + '\n'
        elif self.visualization_type == 1:
            for key in self.notes_dict:
                harmony_tuple = self.notes_dict[key]
                general_string += str(self.midi_loader.inverse_midi_note_dict[harmony_tuple[0]]) + ' ' +str(self.midi_loader.inverse_harmony_dict[harmony_tuple[1]]) + '\n'                                 
        elif self.visualization_type == 2:
            for key in self.notes_dict:
                time_tuple = self.notes_dict[key]
                general_string += str(self.midi_loader.inverse_midi_note_dict[time_tuple[0]]) + ' ' +str(self.midi_loader.inverse_time_notation_dict[time_tuple[1]]) + '\n'                 
        elif self.visualization_type == 3:
            for key in self.notes_dict:
                general_string += str(self.notes_dict[key]) + '\n'                



        return general_string

    def create_dict_and_training_data_from_pickle(self):
        complete_data = cPickle.load(open(self.data_dir, "rb"))

        self.notes_dict = complete_data['notes_dict']

        # transform the data in the positional interpretation for embeddings
        inverse_dict = dict(zip(self.notes_dict.values(), self.notes_dict))

        self.xdata = np.array(list(map(inverse_dict.get, complete_data['x'])))
        self.ydata = np.array(list(map(inverse_dict.get, complete_data['y'])))

        self.vocab_size = len(self.notes_dict)# I am not completely sure how the embeddings matrix is used, but the error I was having
        #was caused because vacab_size was 12, 64 ans so on, but there was notes like 105, so the index was out, i am not sure because
        #in the char original version it was something like 40, i mean it didnt have a max but the number o diffrent chars, maybe it
        #organized it to match, and I sohuld do it but for now I'll use the exact pith representation in midi  
        print self.vocab_size

        return 0

    # with tha change in the x and y creation I have to modify this method,
    # but for now it stays, I will use the pickles for now
    def create_dict_and_training_data(self, tensor):
        #creates a dict with unique elements which can be used to transform to notes

        self.notes_dict = dict(enumerate(sorted(set(self.midi_loader.list_of_vectors_2_vector(tensor)))))

        # transform the data in the positional interpretation for embeddings
        inverse_dict = dict(zip(self.notes_dict.values(), self.notes_dict))

        xdata = [a[:-1] for a in tensor]
        ydata = [a[1:] for a in tensor]

        xdata = self.midi_loader.list_of_vectors_2_vector(xdata)
        ydata = self.midi_loader.list_of_vectors_2_vector(ydata)

        self.xdata = np.array(list(map(inverse_dict.get, xdata)))
        self.ydata = np.array(list(map(inverse_dict.get, ydata)))

        self.vocab_size = len(self.notes_dict)# I am not completely sure how the embeddings matrix is used, but the error I was having
        #was caused because vacab_size was 12, 64 ans so on, but there was notes like 105, so the index was out, i am not sure because
        #in the char original version it was something like 40, i mean it didnt have a max but the number o diffrent chars, maybe it
        #organized it to match, and I sohuld do it but for now I'll use the exact pith representation in midi  
        print self.vocab_size


    def create_batches(self):
        self.num_batches = int(self.xdata.size / (self.batch_size *
                                                   self.seq_length))

        # When the data (tensor) is too small,
        # let's give them a better error message
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        # this line erase the last elements of the dataset in order to have a 
        # perfect multiple of the parameters to have all the batches with the 
        # same size
        self.xdata = self.xdata[:self.num_batches * self.batch_size * self.seq_length]
        self.ydata = self.ydata[:self.num_batches * self.batch_size * self.seq_length]

        self.x_batches = np.split(self.xdata.reshape(self.batch_size, -1),
                                  self.num_batches, 1)
        self.y_batches = np.split(self.ydata.reshape(self.batch_size, -1),
                                  self.num_batches, 1)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0