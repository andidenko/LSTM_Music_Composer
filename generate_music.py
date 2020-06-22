from preprocessing import *
from postprocessing import *
from create_model import *
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import random
import numpy as np
import configparser
import argparse
import os

class Config:
    """
    Class that reads config file.

    Attributes
    ----------
    model_path : str
        Path to model file
    dataset_path : str
        Path to txt file with string melodies
    output_path : str
        Folder where generated melodies will be saved
    seq_len : int
        Length of sequence that is input to neural network that was specified during training
    word_dim : int
        Number of dimensions of vector space that was specified during training
    lstm_layers : int
        Number of lstm layers that was specified during training
    lstm_cells : int
        Number of lstm cells that was specified during training
    dropout : float
        Value of dropout that was specified during training
    bider : bool
        If LSTM layers are bidirectional or not (specified during training)
    state : bool
        If LSTM layers are stateful or not (specified during training)
    input_batch : int, None
        Size of input batch. If it doesn't specified than batch size is set to None
    output_melodies_number : int
        Number of melodies that will be generated
    output_melody_len : int
        Length of melody than will be generated
    """

    def __init__(self, config_path):
        config = configparser.ConfigParser(allow_no_value=True)
        config.read(config_path)
        self.model_path = config['settings']['model_path']
        self.dataset_path = config['settings']['dataset_path']
        self.output_path = config['settings']['output_path']
        self.seq_len = int(config['settings']['seq_len'])
        self.word_dim = int(config['settings']['word_dim'])
        self.lstm_layers = int(config['settings']['lstm_layers'])
        self.lstm_cells = int(config['settings']['lstm_cells'])
        self.dropout = float(config['settings']['dropout'])
        self.bider = config['settings'].getboolean('bider')
        self.state = config['settings'].getboolean('state')
        if config['settings']['input_batch'] == "":
            self.input_batch = None
        else:
            self.input_batch = int(config['settings']['input_batch'])
        self.output_melodies_number = int(config['settings']['output_melodies_number'])
        self.output_melody_len = int(config['settings']['output_melody_len'])


def load_music_model(model_path, vocab_len, seq_len, word_dim, lstm_layers, lstm_cells, drop, bider, state, batch_size):
    """
    Creates model and loads weights.
    Parameters
    ----------
    model_path : str
        Model weights path
    vocab_len : int
        Length of vocabulary (number of unique elements in dataset)
    seq_len : int
        Length of single train sequence
    word_dim : int
        Number of dimensions to represent a word in a space of features using Embedding
    lstm_layers : int
        Number of LSTM layers in network
    lstm_cells : int
        Number of cells in LSTM layers
    drop : int
        Value of dropout
    bider : bool
        If LSTM layers should be bidirectional
    state : bool
        If LSTM layers should be stateful
    batch_size : int
        Batch size

    Returns
    -------
    Sequential
        Loaded model.
    """

    model = create_model(vocab_len, seq_len, word_dim, lstm_layers, lstm_cells, drop, bider, state, batch_size)
    model.load_weights(model_path)
    return model


def generate(seed_notes, length, seq_len, tokenizer, index_word, model):
    """
    Generates melody.

    Parameters
    ----------
    seed_notes : list
        Sequence of notes that are used to start generation.
        If not specified (list is empty) then seed sequence will be generated randomly.
    length : int
        Length of output melody
    seq_len : int
        Length of input sequence that was specified during training
    tokenizer : Tokenizer
        Tokenizer object that was used for dataset tokenization
    index_word : dict
        Dictionary of elements, where key = word_index, value = word_value
    model : Sequential
        Trained model

    Returns
    -------
    list
        List of generated music elements in str representation.
    """

    sliding_window = []
    total_melody = []
    if len(seed_notes) == 0:
        for _ in range(seq_len):
            random_note = ""
            random_offset = ""
            while not random_note.startswith("n") and not random_note.startswith("c"):
                random_note = random.choice(list(index_word.values()))
            while not random_offset.startswith("o") and not random_offset.startswith("t"):
                random_offset = random.choice(list(index_word.values()))
            sliding_window.append(random_note)
            sliding_window.append(random_offset)
    else:
        sliding_window = seed_notes

    i = 0
    while i < length:
        token_list = tokenizer.texts_to_sequences([sliding_window])[0]
        token_list = pad_sequences([token_list], maxlen=seq_len, padding='pre')
        probes = model.predict(token_list, verbose=0)[0]
        predicted = np.random.choice(len(probes), p=probes)

        output_note = index_word[predicted]
        total_melody.append(output_note)
        sliding_window = total_melody[max(0, i - seq_len):]
        i += 1
    return total_melody


def create_data_for_check(melodies, seq_len):
    """"
    Creates dataset (without output) that was used for training.

    Parameters
    ----------
    melodies : list of sequences
        List of encoded melodies
    seq_len : int
        Length of train sequence that was used during training

    Returns
    -------
    list of sequences
        X part of train dataset.
    """
    x_train = []
    for melody in melodies:
        for i in range(len(melody) - 1):
            start_index = max(0, i - seq_len)
            offset = min(i, seq_len) + 1
            x_train.append(melody[start_index:start_index + offset])
    x_train = pad_sequences(x_train, maxlen=seq_len, padding='pre')

    return x_train


def check_for_overfitting(melody, x_train, max_gap):
    """
    Checks model for overfitting.

    Function checks model for overfitting using following heuristic:
    find maximum overlapping between generated model and each train sequence.
    The larger overlap, the more likely that model is overfitted.
    If length of generated melody is bigger than length of train sequence, then
    algorithm will find overlap between each train sequence and each consistent subsequence of generated melody.

    Parameters
    ----------
    melody : list
        Generated melody (label-encoded)
    x_train : list of sequences
        Training dataset (without output)
    max_gap : int
        Number of consistent elements in overlap that can be different

    Returns
    -------
    int
        Length of maximum overlap.

    Raises
    ------
    ValueError
        If melody length is smaller than train sequence length
    """
    if len(melody) < len(x_train[0]):
        raise ValueError("Melody length cannot be smaller than train sequence length")
    max_overlap = 0
    for train_sequence in x_train:
        rlen = len(train_sequence)
        for j in range(0, len(melody) - rlen + 1):
            curr_overlap = 0
            dist = 0
            for k in range(j, j+rlen):
                if train_sequence[k - j] == melody[k]:
                    curr_overlap += 1
                    dist = 0
                else:
                    dist += 1
                if dist >= max_gap:
                    max_overlap = max(max_overlap, curr_overlap)
                    curr_overlap = 0
                    dist = 0
    return max_overlap


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Music generator")
    parser.add_argument("config_path", type=str, help="Path of config file")
    args = parser.parse_args()

    config = Config(args.config_path)

    melodies = melodies_from_txt(config.dataset_path)
    encoded_melodies, tokenizer, vocab_len = tokenize_melodies(melodies)
    model = load_music_model(config.model_path,
                             vocab_len,
                             config.seq_len,
                             config.word_dim,
                             config.lstm_layers,
                             config.lstm_cells,
                             config.dropout,
                             config.bider,
                             config.state,
                             config.input_batch)

    index_word = {index: item for item, index in tokenizer.word_index.items()}
    x_train = create_data_for_check(encoded_melodies, config.seq_len)

    for i in range(config.output_melodies_number):
        print("Melody number {} is being generated...".format(i+1))
        melody = generate([], config.output_melody_len, config.seq_len, tokenizer, index_word, model)
        melody = str_to_melody(melody)
        melody = create_midi(melody)
        save_melody(melody, config.output_path + "/result{}.mid".format(i+1))
        print("Melody number {} saved successfully\n".format(i+1))