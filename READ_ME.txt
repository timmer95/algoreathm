- - - - - - - - - - -
AlgorEAThm (C) v1 2021
READ ME
- - - - - - - - - - -

= = = steps only required once = = = 
step 1. 
install python

step 2. 
Open the (Windows) command prompt
type 
    pip install nltk
    pip install numpy
    pip install pandas
    pip install datetime
    pip install gensim
    pip install sklearn
    pip install matplotlib
    pip install pickle
    pip install statistics
    pip install warnings

step 3.  
Create a folder on local disk (C:\) called 'algoreathm'

step 4. 
In this folder, put the following files from the folder 
(either from GitHub* or Google Drive):
    algoreathm.py
    classification_functions.py
    labelling_functions.py
    preprocessing_functions.py
    random_forest_functions.py
    vectorization_functions.py
    
    BASE_gne_ste_sg.model
    BASE_100131_processed_notes.npy
    BASE_gne_ste_sg_vectors_dict.npy
    BASE_gne_ste_sg_model.sav
    
    NOTEEVENTS_50notes.csv

*The BASE files are too large for GitHub


= = = steps for using AlgorEAThm = = = 
step 1. 
Open the (Windows) command prompt
type < cd C:\algoreathm >

step 2.
type < python algoreathm.py >

step 3.
Choose the action to take

N.B: 
!!! NEVER DELETE THE BASE FILES OR FUNCTION FILES !!!

All the filenames provided should be without the .type. 
    Example: NOTEEVENTS.csv should be provided as NOTEEVENTS

The .csv file with the notes should hold a column called 'TEXT' which contains the notes in string format

For assigning the labels, AlgorEAThm is not case-sensitive.




