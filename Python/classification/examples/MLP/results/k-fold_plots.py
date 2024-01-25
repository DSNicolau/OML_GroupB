import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib
font = {'family' : 'serif',
        'style' : 'italic',
        'size'   : 16}

matplotlib.rc('font', **font)

df = pd.read_csv('/nfs/home/nvasconcellos.it/softLinkTests/Optimizacao/OML_GroupB/Python/classification/examples/MLP/results/MLP_results.csv', sep=';')
print(df)

# Create and save Precision Plot
value = df['Precision']*100
k_folds =  list(range(len(df)))

# Set the color palette
sns.set_palette("BuGn")

# Create a vertical bar plot
plt.figure(figsize=(12, 10))
sns.barplot(x=k_folds, y=value, color='lightblue')

# Add a line for average value
average_value= np.mean(value)
plt.axhline(average_value, color='red', linestyle='--', linewidth=2, label='Average Precision')

# Set plot labels and title
plt.xlabel('K-Fold')
plt.ylabel('Precision')
plt.title('Precision for Different K-Folds')

plt.ylim(70,100)

# Add legend
plt.legend()

result_dir = "/nfs/home/nvasconcellos.it/softLinkTests/Optimizacao/OML_GroupB/Python/classification/examples/MLP/results"

plt.savefig(result_dir + '/CrossValidation_Precision.png')


# Create and save Recall Plot
value = df['Recall']*100
k_folds =  list(range(len(df)))

# Set the color palette
sns.set_palette("BuGn")

# Create a vertical bar plot
plt.figure(figsize=(12, 10))
sns.barplot(x=k_folds, y=value, color='lightblue')

# Add a line for average value
average_value= np.mean(value)
plt.axhline(average_value, color='red', linestyle='--', linewidth=2, label='Average Recall')

# Set plot labels and title
plt.xlabel('K-Fold')
plt.ylabel('Recall')
plt.title('Recall for Different K-Folds')

plt.ylim(40,100)

# Add legend
plt.legend()

result_dir = "/nfs/home/nvasconcellos.it/softLinkTests/Optimizacao/OML_GroupB/Python/classification/examples/MLP/results"

plt.savefig(result_dir + '/CrossValidation_Recall.png')

# Create and save F1 Plot
value = df['F1']*100
k_folds =  list(range(len(df)))

# Set the color palette
sns.set_palette("BuGn")

# Create a vertical bar plot
plt.figure(figsize=(12, 10))
sns.barplot(x=k_folds, y=value, color='lightblue')

# Add a line for average value
average_value= np.mean(value)
plt.axhline(average_value, color='red', linestyle='--', linewidth=2, label='Average F1-Score')

# Set plot labels and title
plt.xlabel('K-Fold')
plt.ylabel('F1-Score')
plt.title('F1-Score for Different K-Folds')

plt.ylim(55,100)

# Add legend
plt.legend()

result_dir = "/nfs/home/nvasconcellos.it/softLinkTests/Optimizacao/OML_GroupB/Python/classification/examples/MLP/results"

plt.savefig(result_dir + '/CrossValidation_F1.png')
