import numpy as np
from .model import Model


class kNN(Model):
    def predict(self, x_test, n_neighbours=3, distance_type="euclidean", chunk_size=1):
        if self.x_train.shape[0] < n_neighbours:
            raise ValueError(
                "The number of neighbours must be less than or equal to the number of training samples!"
            )
        if x_test.shape[1] != self.x_train.shape[1]:
            raise ValueError(
                "The number of features in x_test and x_train must be the same!"
            )

        num_test_samples = x_test.shape[0]

        # Initialize array to store predicted labels
        predicted_labels = np.zeros(num_test_samples)

        # Performing a for loop for all samples is rather slow
        # We tried to use vectorization, however it's impossible to allocate large matrixes (including our dataset)
        # TO overcome this we perform the loop in chunks, in order to reduce the memory usage
        for i in range(0, num_test_samples, chunk_size):
            end_idx = min(i + chunk_size, num_test_samples)

            distances_chunk = self.distance(
                x_test[i:end_idx, :], self.x_train, distance_type
            )

            indices_chunk = np.argsort(distances_chunk, axis=1)[:, :n_neighbours]

            neighbors_labels_chunk = self.y_train[indices_chunk]

            predicted_labels[i:end_idx] = np.round(
                np.mean(neighbors_labels_chunk, axis=1)
            )
            print('one more')
        return predicted_labels
