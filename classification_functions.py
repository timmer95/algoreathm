def classification(vectorized_dict, t, rf_model, fname, printing=False):
    fname = fname + '_predicted_labels'

    predicted_labels = {}

    for note_no, vectors in vectorized_dict.items():
        label = rf_model.predict([vectors])
        if t < label <= 1:
            label = 1
        elif -t <= label <= t:
            label = 0
        else:  # -1 <= pred < -0.5:
            label = -1
        predicted_labels[note_no] = label

    opened_file = open(f'{fname}.txt', 'w')
    for note_no, label in predicted_labels.items():
        if label == 1:
            text = 'is present'
        elif label == 0:
            text = 'cannot be determined'
        else: #if label == -1
            text = 'is absent'
        total_text = f'For note {note_no}, the chance of malnutrition {text}'
        if printing:
            print(total_text)
        total_text = total_text+'\n'
        opened_file.write(total_text)
    opened_file.close()
    print(f'Predicted labels save in {fname}.txt')