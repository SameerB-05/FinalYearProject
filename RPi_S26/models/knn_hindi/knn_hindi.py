import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from scipy.linalg import sqrtm
from sklearn.covariance import LedoitWolf

_HERE  = Path(__file__).parent
_CACHE = _HERE / "cache"

# label mapping: fixed per training label encoder order
_LABEL_MAP = {0: "LIGHT", 1: "PAIN"}


class KNN_Hindi_model:
    def __init__(self):
        self.best_k            = None
        self.cached_cov_matrices = None
        self.y_train           = None

    # ------------------------------------------------------------------
    def load_model(self):
        self.best_k = int(np.load(_CACHE / "best_k.npy"))
        print(f"Loaded best k: {self.best_k}")

        self.cached_cov_matrices = np.load(
            _CACHE / "cov_matrices.npy", allow_pickle=True
        )
        self.y_train = np.load(_CACHE / "labels.npy", allow_pickle=True)
        print("Unique training labels:", set(self.y_train))

    # ------------------------------------------------------------------
    def _riemannian_distance(self, A, B):
        return np.sqrt(np.trace(A + B - 2 * sqrtm(A @ B)).real)

    # ------------------------------------------------------------------
    def _preprocess(self, data):
        """
        data: (samples, time, channels) — windows raw EEG into 256-sample epochs.
        Returns (n_epochs, 256, channels).
        """
        window_size = 256
        stride      = 256
        epochs = []
        for sample in data:
            for i in range(0, sample.shape[0] - window_size + 1, stride):
                epochs.append(sample[i:i + window_size])
        return np.array(epochs)

    # ------------------------------------------------------------------
    def predict(self, data):
        """
        data: (n_epochs, 256, channels)
        Returns integer label (majority vote across epochs).
        """
        print("KNN_Hindi predict — input shape:", data.shape)
        cov_estimator = LedoitWolf()
        predictions   = []

        for sample in data:
            cov_new   = cov_estimator.fit(sample).covariance_
            distances = np.array([
                self._riemannian_distance(cov_new, cov_train)
                for cov_train in self.cached_cov_matrices
            ])
            neighbors = np.argsort(distances)[:self.best_k]
            labels    = self.y_train[neighbors]
            predictions.append(Counter(labels).most_common(1)[0][0])

        final_pred = Counter(predictions).most_common(1)[0][0]
        print("Epoch predictions:", predictions)
        print("Final prediction:", final_pred)
        return final_pred

    # ------------------------------------------------------------------
    def predict_from_folder(self, folder_path):
        """
        Full inference pipeline from a folder containing data.csv.
        Returns predicted word string.
        """
        print("\n=== KNN_Hindi: predicting from folder ===")
        data_path = Path(folder_path) / "data.csv"

        df = pd.read_csv(data_path)
        print("Raw CSV shape:", df.shape)

        # drop timestamp column, keep 32 EEG channels
        data = df.iloc[:, 1:].astype(np.float32).values
        print("After dropping timestamp:", data.shape)

        # clean and normalize
        data = np.nan_to_num(data)
        mean = np.mean(data, axis=0, keepdims=True)
        std  = np.std(data,  axis=0, keepdims=True)
        std[std == 0] = 1
        data = (data - mean) / std

        # segment into 5-sec windows
        window_full = 256 * 5
        samples = []
        for i in range(0, len(data) - window_full + 1, window_full):
            samples.append(data[i:i + window_full])

        if len(samples) == 0:
            raise ValueError("Not enough data to form a 5-sec window")

        samples = np.array(samples)
        print("5-sec windows shape:", samples.shape)

        # epoch into 256-sample windows
        samples = self._preprocess(samples)
        print("Epochs shape:", samples.shape)

        # predict
        pred = self.predict(samples)
        word = _LABEL_MAP.get(pred, str(pred))
        print("Final Predicted Word:", word)
        return word