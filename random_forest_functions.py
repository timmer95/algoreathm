from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from statistics import mean, stdev
import pickle
import numpy as np

thresholds = [round(i, 4) for i in np.arange(0.1, 0.9, 0.025)]

def print_accuracy_plot(model, train_features, test_features, train_labels,
                        test_labels, thresholds, ptitle='Graph', run=0):
    accuracies1 = []
    accuracies2 = []
    max_accuracy = 0
    best_t = 0
    # Predict test labels
    pred_test_labels = model.predict(test_features)
    pred_train_labels = model.predict(train_features)
    for t in thresholds:
        pred_r_test_labels = []
        for label in pred_test_labels:
            if t < label <= 1:
                label = 1
            elif -t <= label <= t:
                label = 0
            else:  # -1 <= pred < -0.5:
                label = -1
            pred_r_test_labels += [label]

        pred_r_train_labels = []
        for label in pred_train_labels:
            if t < label <= 1:
                label = 1
            elif -t <= label <= t:
                label = 0
            else:  # -1 <= pred < -0.5:
                label = -1
            pred_r_train_labels += [label]

        acc2 = accuracy_score(test_labels, pred_r_test_labels)
        acc1 = accuracy_score(train_labels, pred_r_train_labels)
        accuracies1 += [acc1]
        accuracies2 += [acc2]
        both_acc = acc1 + acc2
        if both_acc > max_accuracy:
            max_accuracy = both_acc
            best_t = t

    plt.plot(thresholds, accuracies1, label=f'Acc.train set #{run}')
    plt.plot(thresholds, accuracies2, label=f'Acc. test set #{run}')
    plt.axvline(best_t, c='#cccccc', linestyle='-')
    plt.xlabel('Thresholds', size = 15)
    #plt.tick_params(labelsize=15)
    plt.ylabel('Correctly predicted labels (*100%)', size = 15)
    plt.tick_params(labelsize=15)
    plt.title(label=ptitle, size = 20)
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.show()
    max_acc1 = max(accuracies1)
    max_acc2 = max(accuracies2)
    return best_t, max_acc1, max_acc2

def perfom_rf(vec_lab_dict, base_model_acc, base_model_t, base_model):

    labels = []
    features = []

    for note_no, value in vec_lab_dict.items():
        features += [value[0]]
        labels += [int(value[1])]

    best_t = []
    accuracies_train = []
    accuracies_test = []

    for i in range(5):
        print(f' > Run {i + 1}')
        
        print('Creating Model ...')
        train_features, test_features, train_labels, test_labels = \
            train_test_split(features, labels, test_size=0.25, stratify=labels)

        rf = RandomForestRegressor(n_estimators=1000)

        print('Training Model ...')
        rf.fit(train_features, train_labels)

        t, acc_train, acc_test = print_accuracy_plot(rf,train_features,
                                                test_features,
                                  train_labels,
                            test_labels,thresholds,'Accuracies',run=i+1)
        best_t += [t]
        accuracies_train += [round(acc_train, 3)]
        accuracies_test += [round(acc_test, 3)]
        print('Acc test: ', acc_test)
        i += 1

    new_t = mean(best_t)
    new_acc = mean(accuracies_test)

    print('\nOptimal thresholds: ', round(new_t, 3), '±', round(stdev(best_t),
                                                            3), '\nAccuracies for '
                                                           'training set:',
           round(mean(accuracies_train), 3), '±', round(stdev(accuracies_train), 3),
          '\nAccuracies for test set:', round(new_acc, 3), '±', round(stdev(
        accuracies_test), 3))

    if new_acc > base_model_acc:
        print('The new model performed better than the old model')
    else:
        print('The new model performed worse than the old model')

    save_model = True if input('Save the new model? yes/no\n > ') == 'yes' else False
    if save_model:
        fname = input('Under what name?\n > ')
        pickle.dump(rf, open(f'{fname}_model.sav', 'wb'))
        print(f' < Saved the new Model under the name {fname}_model.sav >')

    replace_model = True if input('Replace the old model? yes/no\n > ') == 'yes' else False
    if replace_model:
        if save_model:
            base_model_name = f'{fname}_model.sav'
        else:
            fname = input('Under what name?\n > ')
            base_model_name = f'{fname}_model.sav'
        base_model = rf
        base_model_acc = new_acc
        base_model_t = new_t
        print(' < Replaced the old Model > ')

    return base_model, base_model_t, base_model_acc, base_model_name




