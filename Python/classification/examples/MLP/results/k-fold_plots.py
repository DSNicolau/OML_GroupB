import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



df = pd.read_csv('Python/classification/examples/MLP/results/MLP_results.csv')

accuracy= df['Accuracy']*100
k_folds =  list(range(len(df)))
data_Accuracy= pd.DataFrame({'K-Fold': k_folds, 'Accuracy': accuracy})



# Set the color palette
sns.set_palette("BuGn")

# Create a vertical bar plot

plt.figure(figsize=(8, 10))
sns.barplot(x='K-Fold', y='Accuracy', data=data_Accuracy, color='lightblue')

# Add a line for average Accuracy
average_Accuracy= np.mean(accuracy)
plt.axhline(average_Accuracy, color='red', linestyle='--', linewidth=2, label='Average Accuracy')

# Set plot labels and title
plt.xlabel('K-Fold')
plt.ylabel('Accuracy')
plt.title('Accuracy for Different K-Folds')


plt.ylim(70,100)


plt.legend()
plt.show()
