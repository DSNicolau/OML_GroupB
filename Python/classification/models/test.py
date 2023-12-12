import numpy as np
from fastdtw import fastdtw
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming your dataset is in the 'data' variable and has shape (num_samples, num_features)
# 'data' should have the timestamp components in the first few columns and features in the remaining columns

# Example data (replace this with your actual data)
data = np.random.rand(100, 10)  # 100 samples, 10 features

# Split the data into features and timestamp components
timestamp_components = data[:, :5]
features = data[:, 5:]

# Choose a target time series for classification
target_series_index = 0
target_series = features[target_series_index]

# Compute DTW distances between the target series and all other series
dtw_distances = [fastdtw(target_series, series)[0] for series in features]

# Combine the distances with the corresponding class labels (assuming you have class labels)
labels = np.random.randint(2, size=len(data))  # Example binary class labels

# Combine distances and labels into a single array
distances_and_labels = np.column_stack((dtw_distances, labels))

# Sort the array based on DTW distances
sorted_distances_and_labels = distances_and_labels[distances_and_labels[:, 0].argsort()]

# Choose the top k neighbors
k = 5
top_k_neighbors = sorted_distances_and_labels[:k, :]

# Perform majority voting for classification
predicted_class = int(np.round(np.mean(top_k_neighbors[:, 1])))

print(f"Predicted Class for the Target Series: {predicted_class}")

# Optionally, you can split your data into training and testing sets and use scikit-learn's KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize kNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=k, metric=fastdtw)

# Fit the model
knn_classifier.fit(X_train, y_train)

# Predict the class for the target series
predicted_class_sklearn = knn_classifier.predict([target_series])[0]

print(f"Scikit-learn Predicted Class for the Target Series: {predicted_class_sklearn}")

# Evaluate the classifier
accuracy = accuracy_score(y_test, knn_classifier.predict(X_test))
print(f"Accuracy of kNN with DTW: {accuracy}")