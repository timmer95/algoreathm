"""
 1 - chance of malnutrition present
 0 - chance of malnutrition cannot be determined
-1 - chance of malnutrition absent
"""

import numpy as np
import pandas as pd

def label_note(note):
    print(note)
    malnutrition = input("Does this note show signs of malnutrition?\
    \n\t P: \tChance of Malnutrition PRESENT\n\t O: \tChance of Malnutrition "
                         "Cannot be Determined\
    \n\t A: \tChance of Malnutrition ABSENT\n\t Type 'quit' to quit\nAnswer: ")
    print()
    return malnutrition

def view_lab_dict(note_dict):
    labels = ['Absent', 'Cannot be Determined', 'Present']
    for key in note_dict.keys():
        if len(note_dict[key]) == 2:
            print(f"{key} : {labels[note_dict[key][1]+1]}")
        else:
            print(f"{key} : No label assigned yet")


def labelling(note_dict, notes, fname):
    try:
        lab_note_dict = np.load(f'{fname}_labelled.npy',
                                allow_pickle=True)[()]
        print('The current labels:')
        view_lab_dict(lab_note_dict)
        do = input('\nTo extend the labels, '
                   'type E. \nTo restart the labelling process, type R.\n > ')
        if do == 'E':
            print('\n > Existing Labels are Extended\n')
        elif do == 'R':
            lab_note_dict = {}
            print('\n > The Labelling Process is Restarted\n')
    except: 
        print('\n > The Labelling Process is Started\n')
        lab_note_dict = {}

    stop = False
    index_to_label = np.array([i for i in notes.index])
    for key, value in note_dict.items():
        if not stop:
            if key not in lab_note_dict:
                index_no = int(key[4:])
                i = np.where(index_to_label == index_no)[0][0]
                print(120*'=')
                print(f"NOTE {index_no}, {i+1} / {len(note_dict)} : ")
                print(20*'- ')
                maln = label_note(note_dict[key])
                if not maln == 'quit':
                    if maln == 'P' or maln == 'p':
                        maln = 1
                        print('Label assigned: Present')
                    elif maln == 'A' or maln == 'a':
                        maln = -1
                        print('Label assigned: Absent')
                    elif maln == 'O' or maln == 'o':
                        maln = 0
                        print('Label assigned: Cannot be Determined')
                    else: 
                        print('WARNING: Invalid Label')
                    lab_note_dict[key] = (value, maln)
                else:
                    stop = True
                i += 1
            else:
                continue
    print('\n= = = = Results = = = =')
    view_lab_dict(lab_note_dict)
    if len(lab_note_dict) > 0:
        np.save(f'{fname}_labelled.npy', lab_note_dict)
        print(f'Labelled dictionary saved as {fname}_labelled.npy')
    print('Program Finished')

def label_notes():
    fname = input('Provide the filename for the csv file with notes:\n > ')
    print('- ' * 20)
    print(f'Reading file {fname}...')
    try:
        notes = pd.read_csv(fname+'.csv', dtype={4: 'str', 5: 'str'})
        print(' < Loading Data finished >')

        print('Processing Data Format ...')
        note_dict = {}
        for ind in notes.index:
            note_no = 'note' + str(ind)
            note_dict[note_no] = notes.loc[ind, 'TEXT']

        labelling(note_dict, notes, fname)
    except Exception as e: print('WARNING: ', e)
