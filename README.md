# Music Composer
This music composer is based on LSTM model.

### Dependencies
To run this project you need following libraries to be installed:
- tensorflow
- keras
- numpy
- h5py
- music21

### config.ini
Config file stores parameters that should be set before music generation. Note that values of some parameters should be the same as during training.
Config file parameters:
- ***model_path*** : Path to model file.
- ***dataset_path*** : Path to txt file with string melodies.
- ***output_path*** :  Folder where generated melodies will be saved.
- ***seq_len*** : Length of sequence that is input to neural network that was specified during training.
- ***word_dim*** : Number of dimensions of vector space that was specified during training.
- ***lstm_layers*** : Number of lstm layers that was specified during training.
- ***lstm_cells*** : Number of lstm cells that was specified during training.
- ***dropout*** : Value of dropout that was specified during training.
- ***bider*** : If LSTM layers are bidirectional or not (specified during training).
- ***state*** : If LSTM layers are stateful or not (specified during training).
- ***input_batch*** : Size of input batch. If it doesn't specified than batch size is set to None. If trained model is stateful, than set input_batch value to 1.
- ***output_melodies_number*** : Number of melodies that will be generated.
- ***output_melody_len*** : Length of melody than will be generated.
