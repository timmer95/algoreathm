"""
Author: Eva Timmer
Script for: detecting the chance of malnutrition using AlgorEAThm
Usage: python algoreathm.py
"""

# Import general modules
import warnings
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Import custom modules
from preprocessing_functions import preprocess_notes
from vectorization_functions import vectorize_notes
from labelling_functions import label_notes
from classification_functions import classification
from random_forest_functions import perfom_rf


# Main
if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # Welcome
    print()
    print(' - *'*6, 'AlgorEAThm', '* - '*6)

    print(' Loading ...')
    base_model_name = 'BASE_gne_ste_sg_model.sav'
    base_model = pickle.load(open(base_model_name, 'rb'))
    base_model_acc = 0.711
    base_model_t = 0.225

    base_vectors_name = 'BASE_gne_ste_sg_vectors_dict.npy'
    base_vectors = np.load(f'{base_vectors_name}',
                           allow_pickle=True)[()]

    print(' Welcome to AlgorEAThm! \
    \n This tool can be used to detect the chance of malnutrition\n '
          'AlgorEAThm \u00a9 2021.v1')

    # Ask for action
    user_wants_to_quit = False
    while not user_wants_to_quit:

        print(' -'*30)
        print('\nChoose from the following options: type')
        print('\t> Detect Malnutrition :    "D"'
              '\n\t> Label Notes :            "L"'
              '\n\t> Pre-Process Notes :      "P"'
              '\n\t> Update Model :           "U"'
              '\n\t> Review Model :           "R"'
              '\n\t> Quit :                   "Q"')
        action = input('Which action is required?\n > ')

        if action == 'P':
            print(' > > > Pre-Process Notes')
            try:
                fname = input(
                    'Provide the filename of the .csv file with '
                    'notes:\n > ')
                preprocess_notes(return_dict=False, fname=fname)
            except KeyboardInterrupt:
                print(' < < < WARNING: Pre-Processing aborted > > > ')
                answer = input('Quit AlgorEAThm? yes/no\n >')
                if answer == 'yes':
                    user_wants_to_quit = True

        elif action == 'D':
            print(' > > > Detect Malnutrition ')
            fname = input('Provide the filename of the .csv file with '
                          'notes:\n > ')
            try: # Try to load the processed_notes_dict
                processed_notes_dict = np.load(f'{fname}_processed.npy',
                                               allow_pickle=True)[()]
            except: # Create the processed_notes_dict
                try: 
                    processed_notes_dict = preprocess_notes(return_dict=True,
                                                    fname=fname)
                except Exception as e: print('WARNING: ', e)
            try: # Try to load the vectorized_dict
                vectorized_dict = np.load(f'{fname}_vectorized.npy',
                                          allow_pickle=True)[()]
            except: # Create the vectorized_dict
                try:
                    vectorized_dict = vectorize_notes(processed_notes_dict,
                                                      base_vectors,
                                                   return_dict=True,fname=fname)
                except Exception as e: print('WARNING: ', e)
            try: 
                show_immediately = True if input('Show the results? yes/no\n > ') == 'yes' else False
                classification(vectorized_dict, base_model_t, base_model,
                                   fname, printing=show_immediately)
            except Exception as e: print('WARNING: ', e)

        elif action == 'L':
            print(' > > > Label Notes')
            label_notes()

        elif action == 'U':
            print(' > > > Update the Model')
            fname = input('Provide the filename of the .csv file with notes '
                          ':\n '
                          '> ')
            try: # Try to load the vectorized_dict
                vectorized_dict = np.load(f'{fname}_vectorized.npy',
                                          allow_pickle=True)[()]
            except: # Create vectorized_dict from processed_notes_dict
                try: # Try to load processed_notes_dict
                    processed_notes_dict = np.load(f'{fname}_processed.npy',
                                                   allow_pickle=True)[()]
                except: # Create processed_notes_dict
                    processed_notes_dict = preprocess_notes(return_dict=True,
                                                            fname=fname)
                vectorized_dict = vectorize_notes(processed_notes_dict,
                                                  base_vectors,
                                            return_dict=True, fname=fname)

            try:
                labelled_dict = np.load(f'{fname}_labelled.npy',
                                allow_pickle=True)[()]
                vectorized_dict = np.load(f'{fname}_vectorized.npy',
                                        allow_pickle=True)[()]
                vec_lab_dict = {}
                for note_no, vectors in vectorized_dict.items():
                    label = labelled_dict[note_no][1]
                    vec_lab_dict[note_no] = (vectors, label)
                
                plt.figure(figsize=(12,7))
                base_model, base_model_t, base_model_acc, base_model_name = perfom_rf(
                    vec_lab_dict, base_model_acc, base_model_t, base_model)
            except Exception as e: print('WARNING: ', e)

        elif action == 'R':
            print('Classification Model: ', base_model_name, base_model)
            print('Threshold: ', base_model_t)
            print('Accuracy: ', base_model_acc)
            print('Vectors: ', base_vectors_name)
            change_model = True if input('Change the Model? yes/no\n > ') == 'yes' else False
            if change_model:
                try: # Change the Classification Model
                    base_model_name = input('Provide the name of the '
                                            'Classification Model:\n '
                                            '> ') + '.sav'
                    base_model = pickle.load(open(base_model_name, 'rb'))
                    base_model_t = input('Set the labelling threshold:\n > ')
                    base_model_acc = input('Set the accuracy:\n > ')
                except Exception as e: 
                    print('WARNING: ', e)
                    print(' > Resetting old settings.')
                    
                try: # Change the vectors embeddings
                    base_vectors_name = input('Provide the name of the Vectors '
                                              'file\n > ') + '.npy'
                    print(f'Loading {base_vectors_name}')
                    base_vectors = np.load(f'{base_vectors_name}',
                                           allow_pickle=True)[()]
                except Exception as e: 
                    print('WARNING: ', e)
                    print(' > Resetting old settings.')

        elif action == 'Q':
            print(' > > > Quit')
            user_wants_to_quit = True



    print('\n\n AlgorEAThm \u00a9 2021.v1\n < < < Goodbye > > > \n')












