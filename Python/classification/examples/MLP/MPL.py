from multiprocessing.sharedctypes import Value
import torch
import copy
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
#from torchvision.models import VGG16_Weights
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from PIL import Image

import math
import numpy as np
import random

import json
import datetime
import sys
sys.path.append('Python/classification/')
from utils import utils
from utils import Generic_Functions as gf
from MotionNet import MotionNet

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    # Configure the GPU devices
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        tf.config.experimental.set_virtual_device_configuration(device, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=50)])

import pickle
import shutil
import os

import optuna

import copy

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

tbf = None # Gloabal Variable to load the TensorBoard_Functions Library

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

class DatasetSequence(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], int(self.y[idx])

def train_model(model, resultsDir, train_dl, val_dl, config_dic):

    checkpoints_dir = resultsDir + '/checkpoints/'
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()

    non_frozen_parameters = [p for p in model.parameters() if p.requires_grad]
    print("non_frozen_parameters:", non_frozen_parameters)
    optimizer = optim.Adam(non_frozen_parameters, lr=config_dic["Hyperparameters"]["learning_rate"], 
                           weight_decay=config_dic["Hyperparameters"]["L2_regularization_factor"])
    scheduler = ReduceLROnPlateau(optimizer, mode=config_dic["ReduceLROnPlateau"]["mode"], factor=config_dic["ReduceLROnPlateau"]["factor"], patience=config_dic["ReduceLROnPlateau"]["patience"],
                                  threshold=config_dic["ReduceLROnPlateau"]["threshold"], threshold_mode=config_dic["ReduceLROnPlateau"]["threshold_mode"],
                                  cooldown=config_dic["ReduceLROnPlateau"]["cooldown"], min_lr=config_dic["ReduceLROnPlateau"]["min_lr"], eps=config_dic["ReduceLROnPlateau"]["eps"],
                                  verbose=config_dic["ReduceLROnPlateau"]["verbose"])

    filepath = checkpoints_dir + 'best_model.pt'
    earlyStopping = gf.EarlyStopping(patience=config_dic["EarlyStopping"]["patience"], verbose=config_dic["EarlyStopping"]["verbose"], delta=config_dic["EarlyStopping"]["delta"], path=filepath, trace_func=print)

    # History
    history = {"train": {"loss": [], "MAE": [], "MAPE": [], "DisAng": [], "Accuracy": []},
                "val": {"loss": [], "MAE": [], "MAPE": [], "DisAng": [], "Accuracy": []}}
    num_epochs = config_dic["Hyperparameters"]["epochs"]
    for epoch in range(num_epochs):
        train_loss = 0
        val_loss = 0

        train_MAE = 0
        val_MAE = 0

        train_MAPE = 0
        val_MAPE = 0

        train_acc = torch.Tensor([0]).to(config_dic["Hardware"]["Device"])
        val_acc = torch.Tensor([0]).to(config_dic["Hardware"]["Device"])

        gradients_dict = {name + "_GradientNorm":0.0 for name, param in model.named_parameters() if param.requires_grad}

        # Training loop
        model.train()

        import time
        start = time.time()
        for x_batch, y_classes in gf.progressbar(train_dl, "Train\tEpoch %d: " % (epoch + 1), "Batches", 40):
            

            x_batch = x_batch.to(config_dic["Hardware"]["Device"])
            # y_batch = y_batch.to(config_dic["Hardware"]["Device"])

            optimizer.zero_grad()
            outputs = model(x_batch)

            # Loss Calculation
            y_batch = torch.zeros(outputs.shape[0], outputs.shape[1]).to(config_dic["Hardware"]["Device"])
            # print("y_classes: " + str(y_classes))
            y_batch[range(0, outputs.shape[0]), y_classes] = 1
            loss = criterion(outputs, y_batch)
            y_pred = torch.argmax(outputs, dim=1)
            acc = torch.sum(y_pred == y_classes.to(config_dic["Hardware"]["Device"]))/outputs.shape[0]
            train_acc += acc

            loss.backward()

            optimizer.step()

            batch_loss = loss.item() # loss per batch

            train_loss += batch_loss

            # MAE Value Calculation
            batch_MAE = torch.mean(torch.abs(y_batch - outputs))
            train_MAE += batch_MAE
            # MAPE Value Calculation
            perc_error_vector = torch.abs(y_batch[y_batch != 0] - outputs[y_batch != 0]) / y_batch[y_batch != 0]
            batch_MAPE = torch.mean(perc_error_vector) # MAPE per batch
            train_MAPE += batch_MAPE

        end = time.time()
        print("Train Epoch Time: " + str(end-start))

        # Validation loop
        model.eval()
        start = time.time()
        with torch.no_grad():
            for x_batch, y_classes in gf.progressbar(val_dl, "Val. \tEpoch %d: " % (epoch + 1), "Batches", 40):
                x_batch = x_batch.to(config_dic["Hardware"]["Device"])
                # y_batch = y_batch.to(config_dic["Hardware"]["Device"])
               
                outputs = model(x_batch)
                # Loss Value Calculation
                y_batch = torch.zeros(outputs.shape[0], outputs.shape[1]).to(config_dic["Hardware"]["Device"])
                # print("y_classes: " + str(y_classes))
                y_batch[range(0, outputs.shape[0]), y_classes] = 1
                loss = criterion(outputs, y_batch)
                y_pred = torch.argmax(outputs, dim=1)
                acc = torch.sum(y_pred == y_classes.to(config_dic["Hardware"]["Device"]))/outputs.shape[0]
                val_acc += acc

                val_loss += loss.item()

                # MAE Value Calculation
                val_MAE += torch.mean(torch.abs(y_batch - outputs))
                # MAPE Value Calculation
                perc_error_vector = torch.abs(y_batch[y_batch != 0] - outputs[y_batch != 0]) / y_batch[y_batch != 0]
                val_MAPE += torch.mean(perc_error_vector)

        end = time.time()
        print("Val Epoch Time: " + str(end-start))

        train_loss /= len(train_dl)
        val_loss /= len(val_dl)
        scheduler.step(val_loss)

        train_MAE /= len(train_dl)
        val_MAE /= len(val_dl)

        train_MAPE /= len(train_dl)
        val_MAPE /= len(val_dl)

        train_acc /= len(train_dl)
        val_acc /= len(val_dl)

        # Save Train and Validation Metrics' History
        train_MAE = train_MAE.cpu().detach().numpy()
        train_MAPE = train_MAPE.cpu().detach().numpy()
        train_acc = train_acc.cpu().detach().numpy()

        val_MAE = val_MAE.cpu().detach().numpy()
        val_MAPE = val_MAPE.cpu().detach().numpy()
        val_acc = val_acc.cpu().detach().numpy()

        history["train"]["loss"].append(train_loss)
        history["train"]["MAE"].append(train_MAE)
        history["train"]["MAPE"].append(train_MAPE)
        history["train"]["Accuracy"].append(train_acc)

        history["val"]["loss"].append(val_loss)
        history["val"]["MAE"].append(val_MAE)
        history["val"]["MAPE"].append(val_MAPE)
        history["val"]["Accuracy"].append(val_acc)

        # Save Train and Validation Metrics
        if config_dic["General"]["TensorBoard"]:
            train_dic = {"loss": train_loss, "MAE": train_MAE, "MAPE": train_MAPE}
            val_dic = {"loss": val_loss, "MAE": val_MAE, "MAPE": val_MAPE}
            if config_dic["Loss"]["Function"] == "MSE":
                train_dic["Accuracy"] = train_acc.item()
                val_dic["Accuracy"] = val_acc.item()
            tbf.write_scalars(config_dic["TensorBoard"]["train_summary_writer"], train_dic, epoch)
            tbf.write_scalars(config_dic["TensorBoard"]["val_summary_writer"], val_dic, epoch)

        print("Current Lr: " + str(optimizer.param_groups[0]["lr"]))

        print(f"Epoch {epoch+1}/{num_epochs}, \tTrain Loss: {train_loss:e},\t\tVal Loss: {val_loss:e}\n\t\tTrain MAE: {train_MAE:e},\t\tVal MAE: {val_MAE:e}")
        print(f"\t\tTrain MAPE: {train_MAPE},\tVal MAPE: {val_MAPE}")
        print(f"\t\tTrain Acc: {train_acc},\tVal Acc: {val_acc}")

        print("first layer:")
        print(list(model.parameters())[1])

        print("last layer:")
        print(list(model.parameters())[-1])

        # Earliy Stopping and Saving best model based on validation loss
        earlyStopping(val_loss, model)

        if earlyStopping.early_stop:
            print("\nEarly stopping!!!")
            break
    with open(resultsDir + '/history.pkl', 'wb') as f:
        pickle.dump(history, f)

    return history, earlyStopping.val_loss_min

def run(config_dic : dict, **kwargs):

    if kwargs.get("batch_size") is not None:
        config_dic["Hyperparameters"]["batch_size"] = kwargs.get("batch_size")

    if kwargs.get("lr") is not None:
        config_dic["Hyperparameters"]["learning_rate"] = kwargs.get("lr")

    if kwargs.get("layer_from_FT") is not None:
        config_dic["Fine_Tuning"]["UnFreezing_Threshold_Layer"] = kwargs.get("layer_from_FT")

    # if kwargs.get("weights") is not None:
    #     config_dic["Fine_Tuning"]["weights"] = kwargs.get("weights")
    # else:
    #     config_dic["Fine_Tuning"]["weights"] = "imagenet"

    if kwargs.get("input_size") is not None:
        config_dic["Hyperparameters"]["input_size"] = kwargs.get("input_size")

    if kwargs.get("hidden_units") is not None:
        config_dic["Hyperparameters"]["hidden_units"] = kwargs.get("hidden_units")

    if kwargs.get("dropout_rate") is not None:
        config_dic["Hyperparameters"]["dropout_rate"] = kwargs.get("dropout_rate")
    else:
        config_dic["Hyperparameters"]["dropout_rate"] = 0.0
        
    if kwargs.get("L2_regularization_factor") is not None:
        config_dic["Hyperparameters"]["L2_regularization_factor"] = kwargs.get("L2_regularization_factor")
    else:
        config_dic["Hyperparameters"]["L2_regularization_factor"] = 0.0

    if kwargs.get("activation") is not None:
        config_dic["Hyperparameters"]["activation"] = kwargs.get("activation")
    else:
        config_dic["Hyperparameters"]["activation"] = "relu"
    
    if kwargs.get("hidden_layers") is not None:
        config_dic["Hyperparameters"]["hidden_layers"] = kwargs.get("hidden_layers")
    else:
        config_dic["Hyperparameters"]["hidden_layers"] = 0

    # Directories Definition
    base_algoritm_dir = config_dic["Directories"]["base_algoritm_dir"]
    if kwargs.get("exp_name") is not None:
        exp_name = kwargs.get("exp_name")
    else:
        exp_name = "MLP_Training_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # exp_name = "VGG16_Prot_Training_" + "Testv2_2"
    resultsDir = base_algoritm_dir + exp_name
    config_dic["General"]["Experiment_Name"] = exp_name

    # Save Configuration
    if not os.path.exists(resultsDir):
        os.makedirs(resultsDir)

    config_dic_json = json.dumps(config_dic, indent=4)
    with open(resultsDir + "/config.json", "w") as f:
        f.write(config_dic_json)

    # Hardware Definition
    device_indx = config_dic["Hardware"]["GPU"]
    avail_devices_count = torch.cuda.device_count()
    actual_device_indx = device_indx if device_indx < avail_devices_count else avail_devices_count - 1
    torch_device = "cuda:" + str(actual_device_indx)

    if config_dic["General"]["TensorBoard"]:
        import tensorflow as tf
        global tbf
        from utils import TensorBoard_Functions as tbf
        # TensorBoard: Set up summary writers
        tensorflowBoardDir = base_algoritm_dir + "/Tensorboard" + "/" + exp_name
        if not os.path.exists(tensorflowBoardDir):
            os.makedirs(tensorflowBoardDir)

        train_log_dir = tensorflowBoardDir + '/train_'
        val_log_dir = tensorflowBoardDir + '/val_'
        summary_log_dir = tensorflowBoardDir + '/markdown/summary'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        config_dic["TensorBoard"] = {"train_summary_writer": train_summary_writer, "val_summary_writer": val_summary_writer}
        # tbf.write_Experiment_Setup(summary_log_dir, config_dic)

        gradients_log_dir = tensorflowBoardDir + '/gradients'
        gradients_summary_writer = tf.summary.create_file_writer(gradients_log_dir)
        config_dic["TensorBoard"]["Gradients_summary_writer"] = gradients_summary_writer


    # Reproducibility (Deterministic) Setup
    def seed_worker(worker_id):
        random.seed(config_dic["Dataset"]["Seed"])
        torch.manual_seed(config_dic["Dataset"]["Seed"])
        torch.cuda.manual_seed(config_dic["Dataset"]["Seed"])
        np.random.seed(config_dic["Dataset"]["Seed"])

    g = torch.Generator()
    g.manual_seed(config_dic["Dataset"]["Seed"])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(True)
    random.seed(config_dic["Dataset"]["Seed"])
    torch.manual_seed(config_dic["Dataset"]["Seed"])
    torch.cuda.manual_seed(config_dic["Dataset"]["Seed"])
    torch.cuda.manual_seed_all(config_dic["Dataset"]["Seed"])
    np.random.seed(config_dic["Dataset"]["Seed"])

    # train_ds = DatasetSequence(inputsDic["train"], labelsDic["train"])
    # val_ds = DatasetSequence(inputsDic["val"], labelsDic["val"])

    X_train, Y_train = kwargs["train_data"]
    X_val, Y_val = kwargs["val_data"]
    train_ds = DatasetSequence(X_train, Y_train)
    val_ds = DatasetSequence(X_val, Y_val)

    train_dl = DataLoader(train_ds, batch_size=config_dic["Hyperparameters"]["batch_size"], shuffle=config_dic["Dataset"]["Shuffle"],
                        num_workers=4,
                        worker_init_fn=seed_worker,
                        generator=g)
    val_dl = DataLoader(val_ds, batch_size=config_dic["Hyperparameters"]["batch_size"],
                        num_workers=4,
                        worker_init_fn=seed_worker,
                        generator=g)


    # # Model Initialization
    # print('config_dic["Fine_Tuning"]["weights"] = ' + config_dic["Fine_Tuning"]["weights"])
    # if config_dic["Fine_Tuning"]["weights"] == "imagenet":
    #     config_dic["Fine_Tuning"]["weights"] = None

    model = MotionNet(input_size=config_dic["Hyperparameters"]["input_size"],
                      hidden_units=config_dic["Hyperparameters"]["hidden_units"], 
                      dropout_rate=config_dic["Hyperparameters"]["dropout_rate"], 
                      activation=config_dic["Hyperparameters"]["activation"],
                      hidden_layers=config_dic["Hyperparameters"]["hidden_layers"])

    # model.eval()
    device = torch.device(torch_device if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    model = model.to(device)
    print(model)

    # Fine Tuning Setup
    if config_dic["Fine_Tuning"]["Fine_Tuning"]:
        gf.freeze_layers(model, config_dic["Fine_Tuning"]["UnFreezing_Threshold_Layer"])

    # Parallel Computing
    if config_dic["General"]["Parallel_Computing"]:
        model = gf.make_data_parallel(model,[1])

    device=next(model.parameters()).device
    config_dic["Hardware"]["Device"] = device
    
    # Print model dtype
    print("Model dtype: ", next(model.parameters()).dtype)

    # Train the Model
    if config_dic["General"]["Train"]:
        history, min_val_loss = train_model(model, resultsDir, train_dl, val_dl, config_dic)
    else:
        # Load History
        with open(resultsDir + '/history.pkl', 'rb') as f:
            history = pickle.load(f)
            min_val_loss = min(history["val"]["loss"])


    if config_dic["General"]["Plot Results"]:
        # Figures Directory
        figuresDir = resultsDir + "/" + "figures"

        # Plot the training and validation loss
        gf.plot_metrics(history, "loss", figuresDir)
        # Plot the training and validation MAE
        gf.plot_metrics(history, "MAE", figuresDir)
        # Plot the training and validation MAPE
        gf.plot_metrics(history, "MAPE", figuresDir)
        # Plot the training and validation DisAng
        gf.plot_metrics(history, "DisAng", figuresDir)

    # Evaluate the model on the test data
    test_x, test_y = kwargs["test_data"]
    test_x = torch.from_numpy(test_x).to(torch.float32)
    
    model.eval()
    outputs = model(test_x.to(device))
    y_pred = torch.argmax(outputs, dim=1).cpu().detach().numpy()

    accuracy = accuracy_score(test_y, y_pred)
    precision = precision_score(test_y, y_pred)
    recall = recall_score(test_y, y_pred)
    f1 = f1_score(test_y, y_pred)
    cohen_kappa = cohen_kappa_score(test_y, y_pred)
    cf_matrix = confusion_matrix(test_y, y_pred)

    displayConfMatrix(cf_matrix, resultsDir)

    with open(resultsDir + '/results.txt', 'w') as f:
        f.write("Test Results: \n\n")
        f.write("Accuracy: " + str(accuracy) + "\n")
        f.write("Precision: " + str(precision) + "\n")
        f.write("Recall: " + str(recall) + "\n")
        f.write("F1: " + str(f1) + "\n")
        f.write("Cohen Kappa: " + str(cohen_kappa) + "\n")


    return max(history["val"]["Accuracy"])

def create_sequences(data_x, data_y, lookback, flatten=False):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(data_x.shape[0]-lookback):
        feature = data_x[i:i+lookback]
        # target = data_y[i+1:i+lookback+1]
        target = data_y[i+lookback-1]
        if flatten:
            feature = feature.flatten()
            target = target.flatten()
        X.append(feature)
        y.append(target)
    X = np.array(X)
    y = np.array(y)
    return torch.from_numpy(X), torch.from_numpy(y)

# def balance_dataset(x, y, rate_pos):
#     print("Before balancing:")
#     print("Shape of x: ", x.shape)
#     print("Shape of y: ", y.shape)
#     pos = x[y[:, -1] == 1]
#     neg = x[y[:, -1] == 0]
#     print("Positive samples: ", pos.shape[0])
#     print("Negative samples: ", neg.shape[0])
#     t_n = neg.shape[0]
#     t_p = pos.shape[0]
#     t_p_new = (t_n * rate_pos) // (1 - rate_pos)
#     reminer_p = t_p_new - t_p
#     times_t_p = int(reminer_p // t_p)
#     reminer_p = int(reminer_p - (t_p*times_t_p))
#     idx = list(range(t_p))*times_t_p + list(range(reminer_p))
#     new_x = np.concatenate((x, pos[idx]))
#     # new_y = np.concatenate((y, np.ones(int(t_p_new - t_p))))
#     new_y = np.concatenate((y, np.ones(int(t_p_new - t_p)).reshape(-1, 1)))
#     print("After balancing:")
#     print("Shape of x: ", new_x.shape)
#     print("Shape of y: ", new_y.shape)
#     pos = new_x[new_y[:, -1] == 1]
#     neg = new_x[new_y[:, -1] == 0]
#     print("Positive samples: ", pos.shape[0])
#     print("Negative samples: ", neg.shape[0])
#     return new_x, new_y

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
    t_p_new = int((t_n * rate_pos) // (1 - rate_pos))
    reminer_p = t_p_new - t_p
    if reminer_p == 0:
        return x, y
    elif reminer_p > 0:        
        times_t_p = int(reminer_p // t_p)
        reminer_p = int(reminer_p - (t_p * times_t_p))
        idx = torch.cat([torch.arange(t_p)] * times_t_p + [torch.arange(reminer_p)])
        new_x = torch.cat((x, pos[idx]))
        new_y = torch.cat((y, torch.ones(int(t_p_new - t_p)).reshape(-1, 1)))
    else:
        reminer_n = -reminer_p
        times_t_n = int(reminer_n // t_n)
        reminer_n = int(reminer_n - (t_n * times_t_n))
        idx = torch.cat([torch.arange(t_n)] * times_t_n + [torch.arange(reminer_n)])
        new_x = torch.cat((x, neg[idx]))
        reminer_n = -reminer_p
        new_y = torch.cat((y, torch.zeros(int(reminer_n)).reshape(-1, 1)))
    
    print("After balancing:")
    print("Shape of x: ", new_x.shape)
    print("Shape of y: ", new_y.shape)
    
    pos = new_x[new_y[:, -1] == 1]
    neg = new_x[new_y[:, -1] == 0]
    print("Positive samples: ", pos.shape[0])
    print("Negative samples: ", neg.shape[0])
    
    return new_x, new_y


def objective(trial):
    global train_x, train_y, val_x, val_y, test_x, test_y
        
    # train_x = (np.random.rand(train_x.shape[0], train_x.shape[1]) - 0.5) * 2
    # train_y = np.random.randint(2, size=train_y.shape[0])
    # val_x = (np.random.rand(val_x.shape[0], val_x.shape[1]) - 0.5) * 2
    # val_y = np.random.randint(2, size=val_y.shape[0])
    
    # seq_length = trial.suggest_int('seq_length', 1, 30, log=False)
    seq_length = 1
    
    # Create sequences for training and validation
    x_train, y_train = create_sequences(train_x, train_y, seq_length, flatten=True)
    x_val, y_val = create_sequences(val_x, val_y, seq_length, flatten=True)
    x_train, x_val = x_train.to(torch.float32), x_val.to(torch.float32)
    
    trial_batch_size = trial.suggest_int("batch_size",512,2048,log=False)
    trial_lr = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
    
    # trial_dropout_rate = trial.suggest_float("dropout_rate", 0.0, 1, log=False)
    # trial_L2_regularization_factor = trial.suggest_float("L2_regularization_factor", 1e-6, 1e-4, log=True)
    # trial_hidden_layers = trial.suggest_int("hidden_layers", 1, 5, log=False)
    # trial_hidden_units = trial.suggest_int("hidden_units", 16, 64, log=False)
    # trial_ActivationFunction = trial.suggest_categorical("ActivationFunction", ["relu", "sigmoid", "tanh"])
    
    trial_dropout_rate = 0.2877382251798777
    trial_L2_regularization_factor = 4.274361601215326e-06
    trial_hidden_layers = 5
    trial_hidden_units = 20
    trial_ActivationFunction = "tanh"
    
    input_size = x_train.shape[1]
    
    # Print hyperparameters
    print("Batch size: ", trial_batch_size)
    print("Learning rate: ", trial_lr)
    print("Input size: ", input_size)
    print("Hidden units: ", trial_hidden_units)
    
    # # Balance dataset
    # print("Balancing dataset...")
    # print("Training set:")
    # x_train, y_train = balance_dataset(x_train, y_train, rate_pos=0.5)
    # print("Validation set:")
    # x_val, y_val = balance_dataset(x_val, y_val, rate_pos=0.5)

    config_dic = json.load(open(config_file_path))
    
    config_dic["Hardware"]["GPU"] = 0
    
    results_dir = "Optuna/%s/Trial_%d" % (studyName, trial.number)

    val_acc = run(config_dic=config_dic, 
                                    batch_size=trial_batch_size, lr=trial_lr,
                                    input_size=input_size, 
                                    hidden_units=trial_hidden_units,
                                    train_data=(x_train, y_train),
                                    val_data=(x_val, y_val),
                                    test_data=(test_x, test_y),
                                    dropout_rate=trial_dropout_rate,
                                    L2_regularization_factor=trial_L2_regularization_factor,
                                    layer_from_FT = None,
                                    hidden_layers = trial_hidden_layers,
                                    activation = trial_ActivationFunction,
                                    exp_name= results_dir)  
    
    return val_acc


# Dictionary with the permutations of 4
idx = {
    0:  [0, 1, 2, 3],
    1:  [0, 1, 3, 2],
    2:  [0, 2, 1, 3],
    3:  [0, 2, 3, 1],
    4:  [0, 3, 1, 2],
    5:  [0, 3, 2, 1],
    6:  [1, 0, 2, 3],
    7:  [1, 0, 3, 2],
    8:  [1, 2, 0, 3],
    9:  [1, 2, 3, 0],
    10: [1, 3, 0, 2],
    11: [1, 3, 2, 0],
    12: [2, 0, 1, 3],
    13: [2, 0, 3, 1],
    14: [2, 1, 0, 3],
    15: [2, 1, 3, 0],
    16: [2, 3, 0, 1],
    17: [2, 3, 1, 0],
    18: [3, 0, 1, 2],
    19: [3, 0, 2, 1],
    20: [3, 1, 0, 2],
    21: [3, 1, 2, 0],
    22: [3, 2, 0, 1],
    23: [3, 2, 1, 0]
}

# loaded_data = utils.load_data_v2()
# train_data, val_data, test_data = loaded_data
# train_x, train_y = train_data
# val_x, val_y = val_data
# test_x, test_y = test_data


# Study iterations idxs 8 studies with 3 idxs each
iters_run = [list(range(i, i+3)) for i in range(0, 24, 3)]

run_number = 0

config_file_path = "/nfs/home/nvasconcellos.it/softLinkTests/Optimizacao/OML_GroupB/Python/classification/examples/MLP/parameters_Offline_GPU0.json"

for cross_validation_iter in iters_run[run_number]:
    subset_idx = idx[cross_validation_iter]

    loaded_data = utils.load_data_v2(cross_validation=True, num_subdivisions=4)
    train_data, val_data, test_data = (np.concatenate((loaded_data[subset_idx[0]], 
                                                    loaded_data[subset_idx[1]]), axis=0), 
                                    loaded_data[subset_idx[2]], 
                                    loaded_data[subset_idx[3]])
    train_x, train_y = train_data[:, :-1], train_data[:, -1]
    val_x, val_y = val_data[:, :-1], val_data[:, -1]
    test_x, test_y = test_data[:, :-1], test_data[:, -1]

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

    studyName = "OML_MLP_CrossValidation_study_%d" % cross_validation_iter

    study = optuna.create_study(
                                # directions=['maximize', 'maximize'],
                                direction='maximize',
                                storage="sqlite:///OML_Database.db",
                                study_name=studyName, load_if_exists=True)

    study.optimize(objective, n_trials=50)

