import sys
sys.path.append('Python/classification/')
import os 

from utils import utils
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    # Configure the GPU devices
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        tf.config.experimental.set_virtual_device_configuration(device, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)])
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import optuna

def plot_metrics(history, resultsModel_dir_):
        
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    num_epochs = len(history.history['loss'])
    epochs_range = range(num_epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.xlabel("Epochs")
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.xlabel("Epochs")
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
        
    figure_name = resultsModel_dir_ + '/LSTM_loss.png'
    plt.savefig(figure_name)
    #plt.show()
    plt.close('all')
    
    f_score_hist = history.history['f_score']
    val_f_score_hist = history.history['val_f_score']
    plt.figure(figsize=(8, 8))
    plt.plot(epochs_range, f_score_hist, label='Training f_score')
    plt.plot(epochs_range, val_f_score_hist, label='Validation f_score')
    plt.xlabel("Epochs")
    plt.legend(loc='lower right')
    plt.title('Training and Validation f_score')
    figure_name = resultsModel_dir_ + '/LSTM_f_score.png'
    plt.savefig(figure_name)
    plt.close('all')
    return figure_name

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
    
def f_score(y_true, y_pred):
    B = 1
    # Calculate TP, TN, FP, FN using tensorflow
    TP = tf.math.count_nonzero(y_pred * y_true)
    TN = tf.math.count_nonzero((y_pred - 1) * (y_true - 1))
    FP = tf.math.count_nonzero(y_pred * (y_true - 1))
    FN = tf.math.count_nonzero((y_pred - 1) * y_true)
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    return (1 + B**2) * (precision * recall) / ((B**2) * precision + recall)
    

data = utils.load_data()
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
    pos = x[y[:, -1] == 1]
    neg = x[y[:, -1] == 0]
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
    # new_y = np.concatenate((y, np.ones(int(t_p_new - t_p))))
    new_y = np.concatenate((y, np.ones(int(t_p_new - t_p)).reshape(-1, 1)))
    print("After balancing:")
    print("Shape of x: ", new_x.shape)
    print("Shape of y: ", new_y.shape)
    pos = new_x[new_y[:, -1] == 1]
    neg = new_x[new_y[:, -1] == 0]
    print("Positive samples: ", pos.shape[0])
    print("Negative samples: ", neg.shape[0])
    return new_x, new_y

def objective(trial):
    
    # Define hyperparameters
    num_hidden_layers = trial.suggest_int('num_hidden_layers', 1, 10, log=False)
    units_lstm_in = trial.suggest_int('units_lstm_in', 16, 256, log=False)
    units_lstm_hidden = trial.suggest_int('units_lstm_hidden', 16, 256, log=False)
    seq_length = trial.suggest_int('seq_length', 1, 30, log=False)
    learning_rate = trial.suggest_float('learning_rate', 1e-8, 1e-1, log=True)
    # focal_alpha = trial.suggest_float('focal_alpha', 0.8, 1, log=True)
    
    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    total = train_y.shape[0]
    neg = (train_y == 0).sum()
    pos = (train_y == 1).sum()
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)

    # class_weight = {0: weight_for_0, 1: weight_for_1}
    class_weight = None

    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))
    
    # Constant parameters
    seed = 123
    batch_size = 2048
    num_epochs = 100
    focal_gamma = 2.0

    # Create sequences for training set
    X_train, y_train = create_sequences(train_x, train_y, seq_length)
    # Create sequences for validation set
    X_val, y_val = create_sequences(val_x, val_y, seq_length)
    # Create sequences for test set
    X_test, y_test = create_sequences(test_x, test_y, seq_length)
    
    # Balance dataset
    print("Balancing dataset...")
    print("Training set:")
    X_train, y_train = balance_dataset(X_train, y_train, rate_pos=0.5)
    print("Validation set:")
    X_val, y_val = balance_dataset(X_val, y_val, rate_pos=0.5)
    print("Test set:")
    X_test, y_test = balance_dataset(X_test, y_test, rate_pos=0.5)

    # Create tensorflow datasets
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    # Shuffle and batch    
    train_ds = train_ds.shuffle(X_train.shape[0], seed=seed).batch(batch_size=batch_size)
    val_ds = val_ds.shuffle(X_val.shape[0], seed=seed).batch(batch_size=batch_size)
    test_ds = test_ds.shuffle(X_test.shape[0], seed=seed).batch(batch_size=batch_size)

    # Check shapes
    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_val shape: ", X_val.shape)
    print("y_val shape: ", y_val.shape)
    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape)

    results_dir = "/nfs/home/nvasconcellos.it/softLinkTests/Optimizacao/OML_GroupB/Python/classification/Examples/LSTM/results/%s/Trial_%d" % (studyName, trial.number)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    import random
    random.seed(seed)

    # Building the LSTM Model
    with tf.device('/device:GPU:1'):
        model = Sequential()        
        model.add(LSTM(units=units_lstm_in, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), 
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=random.randint(0, 1000)), 
                    recurrent_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=random.randint(0, 1000))))
        for i in range(num_hidden_layers):
            model.add(LSTM(units=units_lstm_hidden, return_sequences=True, 
                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=random.randint(0, 1000)), 
                        recurrent_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=random.randint(0, 1000))))
        model.add(Dense(units=1, activation="sigmoid", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=random.randint(0, 1000))))

        optimizer = keras.optimizers.experimental.Adam(learning_rate=learning_rate)
        
        # loss = tf.keras.losses.BinaryFocalCrossentropy(
        #     apply_class_balancing=True,
        #     alpha=focal_alpha,
        #     gamma=focal_gamma,
        #     from_logits=False,
        #     label_smoothing=0.0,
        #     axis=-1,
        # )
        
        # loss = tf.keras.losses.MeanSquaredError()
        
        loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=False,
            label_smoothing=0.0,
            axis=-1,
        )
        
        model.compile(loss=loss, optimizer=optimizer,                         
                        metrics=['accuracy', f_score])
        model.summary()

        checkpoints_dir = results_dir + '/checkpoints/'
        filepath = checkpoints_dir + 'best_model.h5'
        monitor = 'val_f_score'
        mode = 'max'
        checkpoint = ModelCheckpoint(filepath, monitor=monitor, verbose=1, 
                                    save_best_only=True, save_weights_only=False, 
                                    mode=mode, period=1)
        
        # Stop Training, if no improvement observed. (https://keras.io/api/callbacks/early_stopping/)
        Earlystop = EarlyStopping( monitor=monitor, min_delta= 0,patience= 15,verbose=1, mode=mode)
        
        # Reduce learning rate when perf metric stopped improving. (https://keras.io/api/callbacks/reduce_lr_on_plateau/)
        LR = ReduceLROnPlateau(monitor=monitor, factor=0.01, patience=5, cooldown=4, verbose=1,mode=mode,min_delta=0.0001)
        
        callbacks_list = [checkpoint, LR, Earlystop] 

        # history = model.fit(X_train, y_train, validation_data=(X_val, y_val), 
        #                     epochs=20, batch_size=2046, 
        #                     shuffle=False,
        #                     class_weight = {0: 1- focal_alpha, 1: focal_alpha},
        #                     callbacks=callbacks_list, verbose=1)
        
        history = model.fit(train_ds, validation_data=val_ds, 
                            epochs=num_epochs,
                            # class_weight = {0: 1- focal_alpha, 1: focal_alpha},
                            class_weight = class_weight,
                            callbacks=callbacks_list, verbose=1)


        plot_metrics(history, results_dir)
        
        y_pred = model.predict(X_val)
        print("y_pred shape: ", y_pred.shape)
        print("val_y shape: ", val_y.shape)
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
        y_val = np.expand_dims(y_val, axis=-1)

        y_pred = y_pred[:, -1, 0]
        y_val = y_val[:, -1, 0]
        print("y_pred.shape: ", y_pred.shape)
        print("y_val.shape: ", y_val.shape)

        manual_accuracy = np.mean((y_pred == y_val))
        print("Manual accuracy: ", manual_accuracy)
        manual_f_score = f_score(y_val, y_pred)
        print("Manual f_score: ", manual_f_score)

        y_pred = y_pred.reshape((-1))
        y_val = y_val.reshape((-1))
        print("y_pred.shape: ", y_pred.shape)
        print("y_val.shape: ", y_val.shape)
        print("min y_pred: ", np.min(y_pred))
        print("max y_pred: ", np.max(y_pred))
        print("min y_val: ", np.min(y_val))
        print("max y_val: ", np.max(y_val))
        cf_matrix = confusion_matrix(y_val, y_pred)
        displayConfMatrix(cf_matrix, results_dir)
        
        best_f_score = max(history.history['val_f_score'])
        idx_best_f_score = np.argmax(history.history['val_f_score'])
        accuracy = history.history['val_accuracy'][idx_best_f_score]
        # return best_f_score, accuracy
        return accuracy

# studyName = "OML_LSTM_Class_Oversampling_Acc&F1-Score_study_v2"
studyName = "OML_LSTM_Class_Oversampling_Acc_study"

study = optuna.create_study(
                            # directions=['maximize', 'maximize'],
                            direction='maximize',
                            storage="sqlite:////nfs/home/nvasconcellos.it/softLinkTests/xDNN_test.db",
                            study_name=studyName, load_if_exists=True)

study.optimize(objective, n_trials=100)

# # Evaluate the model
# loss, accuracy, f_score_ = model.evaluate(test_ds)
# print("Test loss: ", loss)
# print("Test accuracy: ", accuracy)
# print("Test f_score: ", f_score_)

# y_pred = model.predict(X_test)
# print("y_pred shape: ", y_pred.shape)
# print("test_y shape: ", test_y.shape)
# y_pred[y_pred > 0.5] = 1
# y_pred[y_pred <= 0.5] = 0
# y_test = np.expand_dims(y_test, axis=-1)

# y_pred = y_pred[:, -1, 0]
# y_test = y_test[:, -1, 0]
# print("y_pred.shape: ", y_pred.shape)
# print("y_test.shape: ", y_test.shape)

# manual_accuracy = np.mean((y_pred == y_test))
# print("Manual accuracy: ", manual_accuracy)
# manual_f_score = f_score(y_test, y_pred)
# print("Manual f_score: ", manual_f_score)

# y_pred = y_pred.reshape((-1))
# y_test = y_test.reshape((-1))
# print("y_pred.shape: ", y_pred.shape)
# print("y_test.shape: ", y_test.shape)
# print("min y_pred: ", np.min(y_pred))
# print("max y_pred: ", np.max(y_pred))
# print("min y_test: ", np.min(y_test))
# print("max y_test: ", np.max(y_test))
# cf_matrix = confusion_matrix(y_test, y_pred)
# displayConfMatrix(cf_matrix, results_dir)
