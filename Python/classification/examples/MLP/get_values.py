import os

import numpy as np
import pandas as pd

import optuna


studies_dir = "/nfs/home/nvasconcellos.it/softLinkTests/Optimizacao/OML_GroupB/Python/classification/Examples/MLP/results/Optuna"

results_list = []

for study_num in range(24):
    study_name = "OML_MLP_CrossValidation_study_%d" % study_num

    study = optuna.create_study(
                                # directions=['maximize', 'maximize'],
                                direction='maximize',
                                # storage="sqlite:////nfs/home/nvasconcellos.it/softLinkTests/xDNN_test.db",
                                storage="sqlite:///OML_Database.db",
                                study_name=study_name, load_if_exists=True)

    trials_df = study.trials_dataframe()
    print(trials_df)
    accuracy = trials_df["value"][trials_df["state"] == "COMPLETE"]
    best_trial_idx = np.argmax(accuracy)
    trial_name = "Trial_%d" % best_trial_idx
    trial_dir = os.path.join(studies_dir, study_name, trial_name)
    
    with open(trial_dir + "/results.txt", "r") as f:
        lines = f.readlines()
    
    test_accuracy = float(lines[2].split(":")[1].strip())
    test_precision = float(lines[3].split(":")[1].strip())
    test_recall = float(lines[4].split(":")[1].strip())
    test_f1 = float(lines[5].split(":")[1].strip())
    results_list.append([test_accuracy, test_precision, test_recall, test_f1])
    
results_np = np.array(results_list)
results_df = pd.DataFrame(results_np, columns=["Accuracy", "Precision", "Recall", "F1"])

# Save the results as a CSV file
results_dir = "/nfs/home/nvasconcellos.it/softLinkTests/Optimizacao/OML_GroupB/Python/classification/Examples/MLP/results/"
results_df.to_csv(results_dir + "MLP_results.csv", index=True)
    