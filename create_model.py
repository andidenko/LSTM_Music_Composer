from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.utils import plot_model


def create_model(vocab_len, seq_len, word_dim, lstm_layers, lstm_cells, drop=0.2, bider=False, state=False, batch_size=None):
    """
    Creates Neural Network model for music generation.

    Parameters
    ----------
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
        Neural Network model.
    """
    model = Sequential()

    if batch_size:
        model.add(Embedding(vocab_len, word_dim, input_length=seq_len, batch_input_shape=(batch_size, seq_len)))
    else:
        model.add(Embedding(vocab_len, word_dim, input_length=seq_len))

    if bider:
        for _ in range(lstm_layers - 1):
            model.add(Bidirectional(LSTM(lstm_cells, return_sequences=True, dropout=drop, stateful=state)))
        model.add(Bidirectional(LSTM(lstm_cells, return_sequences=False, dropout=drop, stateful=state)))
    else:
        for _ in range(lstm_layers - 1):
            model.add(LSTM(lstm_cells, return_sequences=True, dropout=drop, stateful=state))
        model.add(LSTM(lstm_cells, return_sequences=False, dropout=drop, stateful=state))

    for _ in range(2):
        model.add(BatchNormalization())
        model.add(Dropout(drop))
        model.add(Dense(256, activation='relu'))

    model.add(BatchNormalization())
    model.add(Dropout(drop))
    model.add(Dense(vocab_len, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model