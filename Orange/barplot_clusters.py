import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Assuming you have a list of silhouette scores for each cluster
silhouette_scores = [0.395, 0.418, 0.462, 0.467, 0.421, 0.383, 0.351, 0.344, 0.347]

# Assuming you have a list of cluster labels
cluster_labels = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# Create a DataFrame for easier plotting
data = pd.DataFrame({'Cluster': cluster_labels, 'Silhouette Score': silhouette_scores})

# Sort the DataFrame by silhouette scores
data = data.sort_values(by='Silhouette Score', ascending=False)

# Set the color palette
# sns.set_palette(sns.color_palette("pastel",6))

# Create a horizontal bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Silhouette Score', y='Cluster', data=data, orient='h',color='tomato')

# Set plot labels and title
plt.xlabel('Silhouette Score')
plt.ylabel('Number of Clusters')
plt.title('Silhouette Scores for K-means Clusters')



# Display the plot
plt.show()