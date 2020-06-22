from music21 import *
import fractions
import re
import os


def str_to_note(s_note):
    """
    Converts string to note.

    Parameters
    ----------
    s_note : str
        String representation of note

    Returns
    -------
    note.Note
        Note of given pitch and quarter length
    """

    data = s_note.split(':')
    return note.Note(data[0], quarterLength=float(data[-1]))


def str_to_chord(s_chord):
    """
    Converts string to chord.

    Parameters
    ----------
    s_chord : str
        String representation of chord

    Returns
    -------
    chord.Chord
        Chord of given pitches and quarter length
    """
    data = s_chord.split(":")
    return chord.Chord(data[0].split("."), quarterLength=float(data[-1]))


def str_to_rest(s_rest):
    """
    Converts string to rest.

    Parameters
    ----------
    s_rest : str
        String representation of rest

    Returns
    -------
    note.Rest
        Rest of given quarter length
    """

    return note.Rest(quarterLength=float(s_rest))


def str_to_tempo(s_tempo):
    """
    Converts string to tempo.

    Parameters
    ----------
    s_tempo : str
        String representation of tempo

    Returns
    -------
    tempo.MetronomeMark
        Tempo of given type
    """

    return tempo.MetronomeMark(number=float(s_tempo))


def str_to_melody(s_melody):
    """
    Converts string melody (list or sting of string representation of music elements) to list of Music21 music objects.

    Parameters
    ----------
    s_melody : str or list
        String representation of melody

    Returns
    -------
    list
        List of Music21 music objects (notes, chords, tempos, etc.)
    """

    if isinstance(s_melody, str):
        s_melody = s_melody.split(" ")

    midi_melody = []
    total_offset = 0
    last_offset = 0.5
    was_offset = False
    for element in s_melody:
        # If element is not offset
        if not element.startswith("o"):
            if not was_offset:
                total_offset += last_offset
                midi_melody.append(total_offset)

            # If element is note
            if element.startswith("n"):
                midi_melody.append(str_to_note(element[1:]))
            # If element is chord
            elif element.startswith("c"):
                midi_melody.append(str_to_chord(element[1:]))
            # If element is tempo
            elif element.startswith("t"):
                midi_melody.append(str_to_tempo(element[1:]))
            # If element is rest
            elif element.startswith("r"):
                midi_melody.append(str_to_rest(element[1:]))

            was_offset = False
        # If element is offset
        else:
            # Skip if previous element was an offset
            if was_offset:
                continue

            of = float(element[1:])
            total_offset += of
            midi_melody.append(total_offset)
            was_offset = True
            last_offset = of

    return midi_melody


def create_midi(melody):
    """
    Converts list of Music21 objects to midi stream.

    Parameters
    ----------
    melody : list
        List of Music21 objects

    Returns
    -------
    stream.Score
        Midi score (stream of notes, chords, tempos, etc. played by piano)

    Raises
    ------
    ValueError
        If no offset was found between music elements
    """

    melody_part = stream.Part()
    melody_part.insert(instrument.Piano())

    melody_score = stream.Score()

    for i in range(1, len(melody), 2):
        offset = melody[i - 1]
        if not (isinstance(offset, float) or isinstance(offset, int) or isinstance(offset, fractions.Fraction)):
            raise ValueError("No offset")
        melody_part.insert(offset, melody[i])

    melody_score.append(melody_part)

    return melody_score


def save_melody(melody, file_path):
    """
    Saves melody to .mid file

    Parameters
    ----------
    melody : stream.Score
        Music stream
    file_path : str
        File path where melody should be stored

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If file directory doesn't exist
    """

    if not os.path.exists(os.path.dirname(file_path)):
        raise FileNotFoundError("File directory doesn't exist")

    melody.write("midi", file_path)


if __name__ == "__main__":
    # from preprocessing import *
    # notes = notes_from_txt(r"C:/Users/Student/Documents/projects/music_generator/beeth/test/appass_1.txt")
    # with open(r"C:\Users\Student\Documents\projects\music_generator\results\test.txt", 'r') as file:
    #   sm = file.read()
    # print(sm)
    # melody = str_to_melody(sm)
    # for e in melody:
    #    print(e)

    n = note.Note('C')
    n.quarterLength = 2
    print(n.duration.components)
    print(n.duration.quarterLength)
    melody = [0.5, n]
    mid = create_midi(melody)
    save_melody(mid, r"C:/Users/Student/Documents/projects/music_generator/new_res/test.mid")
