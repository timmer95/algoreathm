# Vectorization
from gensim.models import Word2Vec, KeyedVectors
import pandas as pd
import numpy as np

def create_own_embedding(dictionary, mname='modelx', dis=5, \
                         minc=1, threads=3, mode='sg', dims=100):
    """Creates own word embedding from dictionary, saves the model

    Args:
        dictionary (dict): a dict holding tokenized and preprocessed sentences
            as a list of lists in the values
        mname (str): the name for the model (default = modelx)
        dim (int): the number of dimensions in the embedding (default = 100)
        dis (int): the distance between a target word and surrounding words \
        (default = 5)
        minc (int): the minimum count of a word's occurrence (default = 1)
        threads (int): the number of workers (default = 3)
        mode (str): the training algorithm, either 'sg' (skip-gram) or
        'cbow' (continuous bag of words) (default = 'sg')

    Returns:
        A tuple of:
            a list of all the numerical vectors
            a list of all the lexical features
    """
    # Create corpus of all the sentences (regardless of the note)
    all_sents = []
    for key in dictionary:
        all_sents += dictionary[key]

    if mode == 'sg':
        m = 1
    elif mode == 'cbow':
        m = 0
    else:
        raise ValueError("Mode should be 'sg' or 'cbow'")

    model = Word2Vec(sentences=all_sents, min_count=minc, window=dis,
                     workers=threads, sg=m, vector_size=dims)
    model.save(f"C:\\thesis_code\\{mname}.model")
    trained_model = Word2Vec.load(f"C:\\thesis_code\\{mname}.model")

    # Create a list of all the numerical vectors
    all_vectors = trained_model.wv[trained_model.wv.index_to_key]

    # Create a list of all the lexical features
    all_features = trained_model.wv.index_to_key

    return trained_model, all_vectors, all_features

def vectorization(use_pretrained=True, sents_dictionary=None,
                  extend_vocab=False,
                  mode='sg', dims = None):
    """Creates a dictionary of features = [vectors]
    Args:
        use_pretrained: Bool, whether to use GoogleNewsEmbeddings
        dictionary: if use_pretrained=False or extend_vocab=True, provide a
        dictionary of { 'note_no': [ [ word, word], [word, word ] ] }  sent
        structure
        extend_vocab: Bool, whether to extend the GNE with STE
        mode: cbow or sg
        dims: provide dimension if use_pretrained=False

    Returns:

    """
    if use_pretrained:
        # Vectorization with Google News Vectors
        filename = r'C:\thesis_code\google_vectors\GoogleNews-vectors-negative300.bin'
        model = KeyedVectors.load_word2vec_format(filename, binary=True)
        feats = model.index_to_key

        # Create a dictionary for indexing the feature vectors
        vectors_dict = dict()
        for key in feats:  # All uniques
            vectors_dict[key] = model.get_vector(key)

        if extend_vocab:
            dims = 300
            model, vecs, feats = create_own_embedding(sents_dictionary,
                                                      mode=mode,
                                                      dims=dims)
            # Create a dictionary for indexing the feature vectors
            for key in feats:  # All uniques
                vectors_dict[key] = model.wv[key]

    else:
        model, vecs, feats = create_own_embedding(sents_dictionary, mode=mode,
                                                  dims=dims)
        # Create a dictionary for indexing the feature vectors
        vectors_dict = dict()
        for key in feats:  # All uniques
            vectors_dict[key] = model.wv[key]

    return vectors_dict

def create_vector_dict(processed_notes, vector_dict_name,
                            use_pretrained=True,
                            extend_vocab=False, mode='sg', dims=100):

    print(f'Settings:\n - - - - - - - ')
    if use_pretrained:
        print('\tWord Embeddings are Google News Vectors')
        print('\tDimensions: 300')
        if extend_vocab:
            print(f'\tWord Embeddings are extended with self-trained '
                  f'embeddings\n\tWords from {len(processed_notes)} '
                  f'notes\n\tAlgorithm: {mode}')
    else:
        print(f'\tWord Embeddings are self-trained embeddings\n\tWords from'
              f' {len(processed_notes)} '
            f'notes\n\tAlgorithm: {mode}')
        print(f'\tDimensions: {dims}')
    print('- '*15)

    print(' > Start < \n')
    print('Vectorization ...')
    print('\t> Training model ...')
    vectors_dict = vectorization(sents_dictionary=processed_notes,
                                 use_pretrained=use_pretrained,
                                 extend_vocab=extend_vocab)

    np.save(f'{vector_dict_name}.npy', vectors_dict)
    print(f'Finished and vectors dict saved as {vector_dict_name}')
    return vectors_dict


def make_feature_vec(words, vectors_dict, dims): # Specify somewhere 100D
    """
    Average the word vectors for a set of words
    """
    feature_vec = np.zeros((dims,), dtype="float32")
    nwords = 0.
    unvectorized_words = 0
    all_words = len(words)

    for word in words:
        if word in vectors_dict:
            nwords = nwords + 1.
            feature_vec = np.add(feature_vec, vectors_dict[word])
        # else:
        #     unvectorized_words += 1
    feature_vec = np.divide(feature_vec, nwords)
    #print(f"{unvectorized_words} / {all_words} words are not found in
    # vocabulary")
    return feature_vec

def get_avg_feature_vecs(notes, vectors_dict):
    """
    Calculate average feature vectors for all reviews

    notes: dict of { note_no (str) : ( [ words (str) ], label (int) ) no sent
    structure
    vectors_dict: dict of { word (str) : np.array([float, float, float])
    dims: number (int) of dimensions. default = 100
    """
    counter = 0
    dims = len(list(vectors_dict.values())[0])
    note_feature_vecs = np.zeros((len(notes),dims), dtype='float32')

    for key, value in notes.items():
        note_feature_vecs[counter] = make_feature_vec(value[0], vectors_dict,
                                                      dims)
        counter = counter + 1
    return note_feature_vecs

def remove_sent_struc(note):
    """list of list of str
    """
    complete_note = []
    for sent in note:
        complete_note += sent
    return complete_note

def create_lab_note_vecs_df(labelled_notes, processed_notes,
                            use_pretrained=True,
                            extend_vocab=False, mode='sg', dims=100):
    """

    notes: dict of { note_no (str) : ( [ [word, word], [word, word] ], label (
    int) ) } sent structure present
    vectors_dict: dict of { word (str) : np.array([float, float, float])
    dims: number (int) of dimensions. default = 100
    mode: cbow or sg,default = 'sg'

    returns: df
    """

    print(f'Settings:\n - - - - - - - ')
    if use_pretrained:
        print('\tWord Embeddings are Google News Vectors')
        print('\tDimensions: 300')
        if extend_vocab:
            print(f'\tWord Embeddings are extended with self-trained '
                  f'embeddings\n\tWords from {len(processed_notes)} '
                  f'notes\n\tAlgorithm: {mode}')
    else:
        print(f'\tWord Embeddings are self-trained embeddings\n\tWords from'
              f' {len(processed_notes)} '
            f'notes\n\tAlgorithm: {mode}')
        print(f'\tDimensions: {dims}')
    print('- '*15)

    print(' > Start < \n')
    print('Removing sentence structure ...')
    notes_without_sents = {}
    for note_no, value in labelled_notes.items():
        notes_without_sents[note_no] = (remove_sent_struc(value[0]), value[1])

    print('Vectorization ...')
    print('\t> Training model ...')
    vectors_dict = vectorization(use_pretrained, processed_notes,
                                 extend_vocab, mode=mode, dims=dims)
    np.save('C:\\algoreathm\\gne_ste_sg_vectors_dict.npy', vectors_dict)
    print('Finished and vecs dict saved')
    """

    print('\t> Vectorize notes ...')
    features = get_avg_feature_vecs(notes_without_sents, vectors_dict)

    print('Creating dataframe ...')
    lab_note_vecs_df= pd.DataFrame(columns=['NOTE_no', 'VECS', 'LABEL'])
    i = 0
    for key, value in notes_without_sents.items():
        lab_note_vecs_df = lab_note_vecs_df.append(
            {'NOTE_no': key, 'VECS': features[i], 'LABEL': value[1]},
            ignore_index=True
        )
        i += 1
    print('\n < Finished >')
    return lab_note_vecs_df
    """

def vectorize_notes(processed_notes_dict, vectors_dict, return_dict=True, \
                                                               fname=None):

    print('Vectorization ...')
    vectors = get_avg_feature_vecs(processed_notes_dict, vectors_dict)

    vectorized_dict = {}
    i = 0
    for key in processed_notes_dict:
        vectorized_dict[key] = vectors[i]
        i += 1

    np.save(f'{fname}_vectorized.npy', vectorized_dict)
    print(f'Dictionary with vectorized notes stored as {fname}_vectorized.npy')
    if return_dict:
        return vectorized_dict