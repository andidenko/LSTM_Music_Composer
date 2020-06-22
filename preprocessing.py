import glob
import sys
import os
from music21 import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.utils as ul
import numpy as np

# List of default values of tempos
tempo_values = [16, 32, 40, 46, 52, 56, 60, 66, 72, 80, 83, 88, 92, 108, 120, 128, 132, 140, 144, 160, 168, 184, 208]


def upper_bound(array, x):
    """
    Finds upper bound index for value x in array.
    Upper bound is the first element in the array which compares greater than value x.

    Parameters
    ----------
    array : tuple, list, 1d-nparray
        Sorted sequence where upper bound is searched
    x : int, float
        Value of the upper bound to search for in the array

    Returns
    -------
    int
        Index to the upper bound position for value x in the array
    """

    n = len(array)
    l = 0
    r = n - 1
    while l < r:
        mid = (l + r) // 2
        if x >= array[mid]:
            l = mid + 1
        else:
            r = mid
    return l


def offset_to_str(offset):
    """
    Converts offset to string and adds prefix 'o' to the result string.

    Parameters
    ----------
    offset : int, float, Fraction
        Value of offset

    Returns
    -------
    str
        String of pattern 'o$' where $ - offset value
        Example: o0.5
    """

    return "o" + str(offset)


def tempo_to_str(element):
    """
    Converts tempo value to string and adds prefix 't' to the result string.

    Tempo value is changed to the upper bound which is in global tempo_values list.
    This is done because tempo from music file can be of any value which can lead to
    creating to many classes. In tempo_values list only default tempos are stored.

    Parameters
    ----------
    element : tempo.MetronomeMark
        Value of tempo

    Returns
    -------
    str
        String of pattern 't$' where $ - tempo value
        Example: t144

    Raises
    ------
    TypeError
        If element type is not tempo.MetronomeMark

    """

    if not isinstance(element, tempo.MetronomeMark):
        raise TypeError("Element type is not Tempo")

    # search for the nearest default tempo value using upper bound
    ind = upper_bound(tempo_values, element.number)
    if tempo_values[ind - 1] == element.number:
        ind -= 1

    return "t" + str(tempo_values[ind])


def note_to_str(element):
    """
    Converts note to string with prefix 'n' and note quarter length.

    Parameters
    ----------
    element : note.Note
        Note to be converted

    Returns
    -------
    str
        String of pattern 'n@:$', where @ - note name, $ - note quarter length
        Example: cD4:1.0

    Raises
    ------
    TypeError
        If element type is not note.Note
    """

    if not isinstance(element, note.Note):
        raise TYpeError("Element type is not Note")

    return "n" + str(element.pitch) + ":" + str(round(float(element.duration.quarterLength), 2))


def chord_to_str(element):
    """
    Converts chord to string with prefix 'c', all notes and chord duration

    Parameters
    ----------
    element : chord.Chord
        Chord to be converted

    Returns
    -------
    str
        String of pattern 'c@:$', where @ - notes' pitches separated by '.', $ - chord quarter length
        Example: cD4.F#4:1.0

    Raises
    ------
    TypeError
        If element type is not chord.Chord
    """

    if not isinstance(element, chord.Chord):
        raise TypeError("Element type is not Chord")

    chord_pitches = list(map(str, element.pitches))
    chord_pitches.sort()
    return "c" + ".".join(chord_pitches) + ":" + str(round(float(element.duration.quarterLength), 2))


def rest_to_str(element):
    """
    Converts rest to string with prefix 'r'

    Parameters
    ----------
    element : note.Rest
        Rest to be converted

    Returns
    -------
    str
        String of pattern 'r$', where $ - rest value
        Example: r1.0

    Raises
    ------
    TypeError
        If element type is not note.Rest
    """

    if not isinstance(element, note.Rest):
        raise TypeError("Element type is not Rest")

    return "r" + str(round(float(element.duration.quarterLength), 2))


def melodies_to_txt(melodies, path):
    """
    Writes parsed melodies to .txt file.

    Parameters
    ----------
    melodies : list
        List of parsed melodies
    path : str
        File path where melodies should be written

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If file path doesn't exist
    """

    if not os.path.exists(os.path.dirname(path)):
        raise FileNotFoundError("File path is not found")

    notes = [" ".join(melody) + "\n" for melody in melodies]
    with open(path, "w") as file:
        file.writelines(notes)


def melodies_from_txt(path):
    """
    Reads melodies from .txt file.

    Parameters
    ----------
    path : str
        Path of .txt file where melodies are stored

    Returns
    -------
    list
        List of melodies in parsed format

    Raises
    ------
    FileNotFoundError
        If file doesn't exist
    """

    if not os.path.exists(path):
        raise FileNotFoundError("File not found")

    with open(path, "r") as file:
        melodies = file.readlines()
    melodies = [melody.split() for melody in melodies]
    return melodies


def get_frequency_table(melodies, element_type):
    """
    Generates frequency table of each element of given type.

    Parameters
    ----------
    melodies : list of sequences
        Melodies where each element is in str format
    element_type : str
        Type of music element: ['note', 'chord', 'tempo', 'offset', 'rest']

    Returns
    -------
    dict
        key - element, value - frequency.

    Raises
    ------
    ValueError
        If element type doesn't exists
    """

    if element_type not in ['note', 'chord', 'tempo', 'offset', 'rest']:
        raise ValueError("Element type doesn't exist")

    element_freq_dict = dict()
    prefix = element_type[0]
    for melody in melodies:
        for element in melody:
            if element.startswith(prefix):
                element_freq_dict[element] = element_freq_dict.get(element, 0) + 1

    return element_freq_dict


def frequency_filter(melodies, element_type, freq_threshold):
    """
    Filter melodies in place by removing elements which frequencies is less than given threshold.

    Removing means replacement of element with low frequency with element with high frequency,
    according to its probability of occurrance.

    Parameters
    ---------
    melodies : list of sequences
        Melodies where each element is in str format
    element_type : str
        Type of music element: ['note', 'chord', 'tempo', 'offset', 'rest']
    freq_threshold : int
        Frequency threshold

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If element type doesn't exists or invalid threshold value was set
    """

    if element_type not in ['note', 'chord', 'tempo', 'offset', 'rest']:
        raise ValueError("Element type doesn't exist")

    if freq_threshold < 1:
        raise ValueError("Invalid threshold value. Threshold should be bigger than 0")

    element_freq_dict = get_frequency_table(melodies, element_type)

    frequency_table = [item for item in element_freq_dict.items()]
    high_frequency_elements, high_frequencies = zip(
        *[element for element in frequency_table if element[1] > freq_threshold])

    if len(high_frequency_elements) == 0:
        raise ValueError("Too high threshold was chosen")

    total_sum = sum(high_frequencies)

    probes = list(map(lambda x: x / total_sum, high_frequencies))

    low_frequency_elements = [element[0] for element in frequency_table if element[1] <= freq_threshold]

    for melody in melodies:
        for idx in range(len(melody)):
            if melody[idx] in low_frequency_elements:
                melody[idx] = np.random.choice(high_frequency_elements, p=probes)


def midi_to_str(folder):
    """
    Parses melodies from folder

    Parameters
    ----------
    folder : str
        Folder path where .mid melodies are stored

    Returns
    -------
    list
        List of parsed melodies

    Raises
    ------
    FileNotFoundError
        If folder path doesn't exist
    """

    if not os.path.exists(folder):
        raise FileNotFoundError("Folder doesn't exists")

    melodies = []
    for file in glob.glob(folder + "/*.mid"):
        melody = converter.parse(file)

        parts = instrument.partitionByInstrument(melody)
        if parts:
            music_elements = parts[0].recurse() # music_elements are notes, chords, tempos, etc.
        else:
            music_elements = melody.flat
        
        str_melody = []
        str_element = ""
        offset = 0
        for element in music_elements:
            if isinstance(element, note.Note):
                str_element = note_to_str(element)
            elif isinstance(element, chord.Chord):
                str_element = chord_to_str(element)
            elif isinstance(element, note.Rest):
                str_element = rest_to_str(element)
            elif isinstance(element, tempo.MetronomeMark):
                if len(str_melody) > 1 and str_melody[-2].startswith("t"):  # Skip repetitive tempos
                    continue
                str_element = tempo_to_str(element)
            else:
                continue
            str_melody.append(offset_to_str(round(float(element.offset - offset), 2)))
            offset = element.offset
            str_melody.append(str_element)
        
        melodies.append(str_melody)

    return melodies


def tokenize_melodies(melodies):
    """
    Encodes each element in parsed melodies with integer value between 1 and vocab_size.

    Parameters
    ----------
    melodies : list
        List of parsed melodies

    Returns
    -------
    list, Tokenizer, int
        Function returns tuple of encoded melodies, tokenizer that was fitted on parsed melodies and vocabulary size
    """

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(melodies)

    total_notes = len(tokenizer.word_index) + 1

    one_hot_melodies = tokenizer.texts_to_sequences(melodies)

    return one_hot_melodies, tokenizer, total_notes


def create_data(melodies, seq_len):
    """
    Creates train dataset from encoded melodies that is used to train Neural Network.

    X is the sequence of elements and y is the element that should be predicted.
    Sliding window technique is used.
    Example:
    If whole sequence is [4, 2, 1, 3, 5, 2] and X sequence lenght is 3, then:
    X           y
    [4, 2, 1]   [2]
    [2, 1, 3]   [5]
    [1, 3, 5]   [2]

    Then y is encoded to one-hot vectors

    Parameters
    ----------
    melodies : list
        List of parsed melodies

    Returns
    -------
    list, Tokenizer, int
        Function returns tuple of encoded melodies, tokenizer that was fitted on parsed melodies and vocabulary size
    """

    x_train = []
    y_train = []
    for melody in melodies:
        for i in range(len(melody) - 1):
            start_index = max(0, i - seq_len)
            offset = min(i, seq_len) + 1
            x_train.append(melody[start_index:start_index + offset])
            y_train.append(melody[start_index + offset])

    x_train = pad_sequences(x_train, maxlen=seq_len, padding='pre')

    y_train = ul.to_categorical(y_train)

    return x_train, y_train