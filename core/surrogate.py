import numpy as np
import os
import joblib

from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF, ConstantKernel as C

class BayesianSurrogate:
    def __init__(self, max_samples: int = 2000):
        kernel = (
            C(1.0, (1e-3, 1e3)) * Matern(
                nu=2.5, 
                length_scale=1.0, 
                length_scale_bounds=(1e-2, 1e3)
            ) + WhiteKernel(
                noise_level=0.1, 
                noise_level_bounds=(1e-5, 1.0)
            )
        )

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.1,
            n_restarts_optimizer=10,
            normalize_y=True
        )

        self.max_samples = max_samples
        self.X = []
        self.y = []
        self.is_fitted = False

    @property
    def n_samples(self):
        return len(self.y)

    def add_observation(self, features, fitness):
        feat_array = np.asarray(features, dtype=np.float32).flatten()

        if self.n_samples >= self.max_samples:
            self.X.pop(0)
            self.y.pop(0)

        self.X.append(feat_array)
        self.y.append(float(fitness))

        if self.n_samples >= 10 and self.n_samples % 5 == 0:
            self.fit()

    def fit(self):
        if self.n_samples < 5:
            return

        X_train = np.asarray(self.X, dtype=np.float32)
        y_train = np.asarray(self.y, dtype=np.float32)

        try:
            self.model.fit(X_train, y_train)
            self.is_fitted = True
        except Exception as e:
            print(f" [Surrogate] Erro ao ajustar GP: {e}")

    def predict(self, features):
        if not self.is_fitted:
            return None, None

        feat_query = np.asarray(features, dtype=np.float32).reshape(1, -1)

        mean, std = self.model.predict(feat_query, return_std=True)

        return float(mean[0]), float(std[0])

    def save(self, path: str = "results/surrogate.joblib"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "model": self.model,
            "X": self.X,
            "y": self.y,
            "is_fitted": self.is_fitted,
            "max_samples": self.max_samples
        }
        joblib.dump(data, path)
        print(f" Surrogate Bayesiano salvo em {path}")

    def load(self, path: str = "results/surrogate.joblib"):
        if os.path.exists(path):
            try:
                data = joblib.load(path)
                self.model = data["model"]
                self.X = data["X"]
                self.y = data["y"]
                self.is_fitted = data["is_fitted"]
                self.max_samples = data.get("max_samples", 2000)
                print(f" Surrogate Bayesiano carregado de {path}")
                return True
            except Exception as e:
                print(f" Erro ao carregar surrogate: {e}")
        return False


def expected_improvement(mean, std, best, xi=0.01):
    """
    Calcula a Melhoria Esperada (EI). 
    Ajuda o motor de busca a equilibrar Exploração vs Explotação.
    """
    if std is None or std <= 1e-9:
        return 0.0

    improvement = mean - best - xi
    Z = improvement / std

    ei = (improvement * norm.cdf(Z) + std * norm.pdf(Z))

    return float(max(ei, 0.0))