import sys
sys.path.append('Python/classification/')
import os 

from utils import utils

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

import optuna

def displayConfMatrix(cf_matrix, resultsModel_dir_): 
    group_names = ['True Neg','False Pos','False Neg', 'True Pos']

    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                        cf_matrix.flatten()/np.sum(cf_matrix)]
    # group_percentages = []
    
    for i in range(cf_matrix.shape[0]):
        for value in (cf_matrix[i].flatten()/np.sum(cf_matrix[i])):
            group_percentages.append("{0:.2%}".format(value))

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(2,2)

    plt.figure()
    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

    ax.set_title('Confusion Matrix with labels\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['Non-Motion', 'Motion'])
    ax.yaxis.set_ticklabels(['Non-Motion', 'Motion'])
    
    # # Create the results directory
    # resultsModel_dir = folder_dir + '/results/' + model_name 
    # if not os.path.exists(resultsModel_dir):
    #     os.mkdir(resultsModel_dir)

    ## Display the visualization of the Confusion Matrix.  
    figure_name = resultsModel_dir_ + '/Confution_Matrix.png'
    plt.gcf().set_size_inches(8, 6)
    plt.savefig(figure_name)

data = utils.load_data_v2()
train_data, val_data, test_data = data
train_x, train_y = train_data
val_x, val_y = val_data
test_x, test_y = test_data

# Remove the time information from the data
train_x = train_x[:, 5:]
val_x = val_x[:, 5:]
test_x = test_x[:, 5:]

# Normalize the data
min_features = train_x.min(axis=0)
max_features = train_x.max(axis=0)

train_x = (train_x - min_features) / (max_features - min_features)
val_x = (val_x - min_features) / (max_features - min_features)
test_x = (test_x - min_features) / (max_features - min_features)

# Sequences creation
def create_sequences(data_x, data_y, seq_length):
    x = []
    y = []
    for i in range(data_x.shape[0] - seq_length):
        x.append(data_x[i:i+seq_length])
        y.append([data_y[i+seq_length - 1]])
        # y.append(data_y[i+1:i+seq_length+1])
    return np.array(x), np.array(y)

def balance_dataset(x, y, rate_pos):
    print("Before balancing:")
    print("Shape of x: ", x.shape)
    print("Shape of y: ", y.shape)
    pos = x[y == 1]
    neg = x[y == 0]
    print("Positive samples: ", pos.shape[0])
    print("Negative samples: ", neg.shape[0])
    t_n = neg.shape[0]
    t_p = pos.shape[0]
    t_p_new = (t_n * rate_pos) // (1 - rate_pos)
    reminer_p = t_p_new - t_p
    times_t_p = int(reminer_p // t_p)
    reminer_p = int(reminer_p - (t_p*times_t_p))
    idx = list(range(t_p))*times_t_p + list(range(reminer_p))
    new_x = np.concatenate((x, pos[idx]))
    new_y = np.concatenate((y, np.ones(int(t_p_new - t_p))))
    # new_y = np.concatenate((y, np.ones(int(t_p_new - t_p)).reshape(-1, 1)))
    print("After balancing:")
    print("Shape of x: ", new_x.shape)
    print("Shape of y: ", new_y.shape)
    pos = new_x[new_y == 1]
    neg = new_x[new_y == 0]
    print("Positive samples: ", pos.shape[0])
    print("Negative samples: ", neg.shape[0])
    return new_x, new_y

# seq_length = 1

# # Create sequences for training set
# X_train, y_train = create_sequences(train_x, train_y, seq_length)
# # Create sequences for validation set
# X_val, y_val = create_sequences(val_x, val_y, seq_length)
# # Create sequences for test set
# X_test, y_test = create_sequences(test_x, test_y, seq_length)

X_train, y_train = train_x, train_y
X_val, y_val = val_x, val_y
X_test, y_test = test_x, test_y

# # Balance dataset
# print("Balancing dataset...")
# print("Training set:")
# X_train, y_train = balance_dataset(X_train, y_train, rate_pos=0.5)
# print("Validation set:")
# X_val, y_val = balance_dataset(X_val, y_val, rate_pos=0.5)
# print("Test set:")
# X_test, y_test = balance_dataset(X_test, y_test, rate_pos=0.5)


def objective(trial):
    
    trial_num_estimators = trial.suggest_int("n_estimators", 10, 500)
    trial_max_depth = trial.suggest_int("max_depth", 1, 100)

    clf = RandomForestClassifier(random_state=123, n_estimators=trial_num_estimators, max_depth=trial_max_depth)

    clf.fit(X_train, y_train)

    # Train Accuracy
    train_y_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_y_pred)
    print("Train Accuracy:", train_accuracy)

    val_y_pred = clf.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_y_pred)
    print("Validation Accuracy:", val_accuracy)
    
    # Test Results
    resultsDir = "/nfs/home/nvasconcellos.it/softLinkTests/Optimizacao/OML_GroupB/Python/classification/examples/RandomForest/results/%s/Trial_%d" % (studyName, trial.number)
    if not os.path.exists(resultsDir):
        os.makedirs(resultsDir)    
    
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(test_y, y_pred)
    precision = precision_score(test_y, y_pred)
    recall = recall_score(test_y, y_pred)
    f1 = f1_score(test_y, y_pred)
    cohen_kappa = cohen_kappa_score(test_y, y_pred)
    cf_matrix = confusion_matrix(test_y, y_pred)
    
    # Print Test Results
    print("Test Results:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("Cohen Kappa:", cohen_kappa)
    
    displayConfMatrix(cf_matrix, resultsDir)

    with open(resultsDir + '/results.txt', 'w') as f:
        f.write("Test Results: \n\n")
        f.write("Accuracy: " + str(accuracy) + "\n")
        f.write("Precision: " + str(precision) + "\n")
        f.write("Recall: " + str(recall) + "\n")
        f.write("F1: " + str(f1) + "\n")
        f.write("Cohen Kappa: " + str(cohen_kappa) + "\n")
        
    # return train_accuracy, val_accuracy
    return accuracy
        
studyName = "OML_RandomForest_Acc_study_Test_v2"

study = optuna.create_study(
                            # directions=['maximize', 'maximize'],
                            direction='maximize',
                            storage="sqlite:///OML_Database.db",
                            study_name=studyName, load_if_exists=True)

study.optimize(objective, n_trials=100)