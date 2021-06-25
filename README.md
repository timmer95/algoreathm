# AlgorEAThm
_Basic Explanation_
AlgorEAThm is a model used for detecting the chance of malnutrition for notes from Electronic Health Records. It provides classification of notes into "chance of malnutrition is present", "chance of malnutrition is absent" and "chance of malnutrition cannot be determined". AlgorEAThm can be used to classify notes, but also to assign labels to notes with a user-friendly interface. Moreover, separately performing pre-processing is possible which might be handy for analysing large files. The model can be updated and the settings reviewed. The initial accuracy of AlgorEAThm is 71.1%.
_Advanced Explanation_
The notes are being preprocessed using NLTK and vectorized with Word2Vec from gensim. For classification AlgorEAThm uses Random Forest from scikit-learn. The settings for the model are BASE_gne_ste_sg_model.sav as Random Forest Classifier (trained on 150 labelled notes, 1000 trees) and BASE_gne_ste_sg_vectors_dict.npy as numerical 300D vectors for the lexical words (Word2Vec, trained on 100131 preprocessed notes with GNE_STE_SG (Google News Embeddings extended with Self-Trained Embeddings using the Skip-Gram algorithm)). The choice for the GNE_STE_SG embeddings was based on own research, which concluded that GNE_STE_SG embeddings performed slightly better than only Google New Embeddings, Self-Trained Embeddings using the Skip-Gram algorithm, Self-Trained Embeddings using the Continuous-Bag-Of-Words algorithm or the Google News Embeddings extended with Self-Trained Embeddings using the Continuous-Bag-Of-Words. The intial AlgorEAThm model with BASE_gne_ste_sg_model.sav and BASE_gne_ste_sg_vectors_dict.npy provides an accuracy of 71.1%. Random Forest assigns a value in the range (-1, 1) to each note and a labelling threshold θ in the range (0, 1)

- - - - - - - - - - -
AlgorEAThm (C) 2021.v1
READ ME
- - - - - - - - - - -

## AlgorEAThm Set-Up
= = = steps only required once = = = 

STEP 1. 

install python
download .exe file from https://www.python.org/downloads/
https://docs.python.org/3.6/using/windows.html#installing-without-ui
Open (Windows) command prompt & type (replace "python-3.6.0.exe" with your .exe file)

    python-3.6.0.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
    


STEP 2. 
In the (Windows) command prompt, type 

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

STEP 3. 

Create a folder on local disk (C:\) called 'algoreathm'

STEP 4. 

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

*The BASE files are too large for GitHub so download them from Google Drive: https://drive.google.com/drive/folders/1CdAq0QOKlLU8hci37RL1eqTTJ1YxMYTE?usp=sharing

- - - - - - - - - - - - - - - - - - - - - 

## AlgorEAThm Usage
= = = steps for using AlgorEAThm = = = 

STEP 1.

Open the (Windows) command prompt & type 

    cd C:\algoreathm 

STEP 2. 

type 

    python algoreathm.py 

STEP 3.

Choose the action to take

< < N.B: > >  
Don't delete the BASE files or function files, the model won't work anymore. However you can re-download them from GitHub/Google Drive. 

All the filenames provided should be without the .type. 
    Example: NOTEEVENTS.csv should be provided as NOTEEVENTS

The .csv file with the notes should hold a column called 'TEXT' which contains the notes in string format

For assigning the labels, AlgorEAThm is not case-sensitive.

