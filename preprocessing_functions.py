"""
Author: Eva Timmer
Script for: Pre-Processing a sheet in the format NOTEEVENTS.csv
Usage: python algoreathm.py NOTEEVENTS00000.csv
"""

from datetime import datetime as dt
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd

# Settings
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Pre-Processing functions
def compose_sents(list_of_words):
    """ Performs sentence segmentation
    Args:
        list_of_all: list of tokenized words including punctuation (str)

    Returns:
        list of list of str: list of sentences, which are lists of words and
        punctuation (str)
    """
    sentences = []  # Initiate list of sentences
    num_sentences = list_of_words.count('.')  # Count the number of sentences

    while num_sentences > 0:  # Repeat while sentences remain
        # Index the end of the sentence by indexing the period
        end = list_of_words.index('.') + 1

        # Add the sentence to the list of sentences
        sentences += [list_of_words[:end]]

        # Remove the sentence from the list of words
        list_of_words = list_of_words[end:]

        num_sentences -= 1  # Subtract the sentence from the number of sentences

    # Check whether the last part of the note is finished with a period
    if len(list_of_words) > 0:
        # Add the remainder of the note as 1 sentence
        end = 0
        sentences += [list_of_words[end:]]

    return sentences


def words_preprocessing(sent):
    words_p = [word.lower() for word in sent if word.isalpha()]
    words_s = [w for w in words_p if not w in stop_words]
    words_lm = [lemmatizer.lemmatize(w) for w in words_s]
    return words_lm


def note_preprocessing(note):
    """
    Preprocesses a note:

    Args:
        note (str): patient note

    Returns:
        list of list of str: 1 list = note, lists = sentences, words = string
            tokenized, lowercased, broken into sentences
            punctuation & stopword removed and lemmatized words
    """
    words = word_tokenize(note)
    # words_l = list(map(str.lower, words))
    # if sent_struc:
    sents = compose_sents(words)
    text = list(map(words_preprocessing, sents))
    # else:
    # text = list(words_preprocessing(words_l))
    return text


def preprocessing(notes):
    start_time = dt.now()
    print('\n > > > Start Pre-Processing < < <\n')

    # Initiate dict of { note_no (str) : [ [ words (str) ], [ words (str) ] ] }
    note_dict = {}
    for i in range(len(notes)):
        print(f'Preprocessing note {i} ...')
        exec(
            f'note_dict["note{i}"] = note_preprocessing(notes.loc[i, "TEXT"])')
    end_time = dt.now()
    run_time = end_time - start_time
    print('Running time: ', run_time)
    print('\n < < < Pre-Processing finished > > >\n')
    return note_dict

def preprocess_notes(return_dict=False, fname=None):
    print('> > > Pre-Process Notes')

    print('- '*20)
    print(f'Reading file {fname}...')
    notes = pd.read_csv(fname+'.csv', dtype={4:'str', 5:'str'})
    print(' < Loading Data finished >')

    print('Pre-Processing data ...')
    processed_notes_dict = preprocessing(notes)

    np.save(f'{fname}_processed.npy', processed_notes_dict)
    print(f'Dictionary with processed notes stored as {fname}_processed.npy')

    if not return_dict:
        return
    else:
        return processed_notes_dict
