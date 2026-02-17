import numpy as np
import pickle

class FitnessNeuralNet:
    def __init__(self, lr=0.001, n_iters=5000, hidden_sizes=[32, 24, 16]):
        self.lr = lr
        self.n_iters = n_iters
        self.hidden_sizes = hidden_sizes
        self.weights = []
        self.biases = []
        self.mean_X = None
        self.std_X = None

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        for i in range(X.shape[1]):
            col_mean = np.nanmean(X[:, i])
            X[np.isnan(X[:, i]), i] = col_mean

        self.mean_X = np.mean(X, axis=0)
        self.std_X = np.std(X, axis=0) + 1e-8
        X = (X - self.mean_X) / self.std_X
        X = np.clip(X, -5, 5)

        n_samples, n_features = X.shape
        n_outputs = y.shape[1]

        layers = [n_features] + self.hidden_sizes + [n_outputs]
        self.weights = [np.random.randn(layers[i], layers[i+1]) * 0.005 for i in range(len(layers)-1)]
        self.biases = [np.zeros((1, layers[i+1])) for i in range(len(layers)-1)]

        for iter in range(self.n_iters):
            a = X
            activations = [X]
            for i in range(len(self.weights) - 1):
                z = a @ self.weights[i] + self.biases[i]
                z = np.clip(z, -15, 15)
                a = np.tanh(z) if i % 2 == 0 else np.sinh(z)
                a = np.clip(a, -10, 10)
                activations.append(a)

            y_pred = a @ self.weights[-1] + self.biases[-1]
            y_pred = np.clip(y_pred, -100, 100)

            error = y_pred - y
            deltas = [error]

            for i in reversed(range(len(self.weights) - 1)):
                if i % 2 == 0:
                    delta = deltas[0] @ self.weights[i+1].T * (1 - activations[i+1]**2)
                else:
                    delta = deltas[0] @ self.weights[i+1].T * np.cosh(activations[i+1])
                    delta = np.clip(delta, -50, 50)
                deltas.insert(0, delta)

            for i in range(len(self.weights)):
                dw = (1 / n_samples) * activations[i].T @ deltas[i]
                db = (1 / n_samples) * np.sum(deltas[i], axis=0, keepdims=True)
                self.weights[i] -= self.lr * dw
                self.biases[i] -= self.lr * db

            if iter % 1000 == 0:
                loss = np.mean(np.abs(error))
                print(f"Iter {iter}: loss = {loss:.4f}")

    def predict(self, X):
        X = np.array(X, dtype=float).reshape(1, -1)

        for i in range(X.shape[1]):
            if np.isnan(X[0, i]):
                X[0, i] = self.mean_X[i] if self.mean_X is not None else 0.0

        if self.mean_X is not None and self.std_X is not None:
            X = (X - self.mean_X) / self.std_X
        X = np.clip(X, -5, 5)

        a = X
        for i in range(len(self.weights) - 1):
            z = a @ self.weights[i] + self.biases[i]
            z = np.clip(z, -15, 15)
            a = np.tanh(z) if i % 2 == 0 else np.sinh(z)
            a = np.clip(a, -10, 10)

        if not self.weights or not self.biases:
            return np.zeros(8)

        y_pred = a @ self.weights[-1] + self.biases[-1]
        return y_pred[0]

    def save_model(self, filename="models/trained_fitness_model.pkl"):
        if not self.weights:
            print("Модель не навчена")
            return

        model_state = {
            'weights': self.weights,
            'biases': self.biases,
            'mean_X': self.mean_X,
            'std_X': self.std_X
        }

        with open(filename, 'wb') as f:
            pickle.dump(model_state, f)
        print(f"Модель збережено в {filename}")

    def load_model(self, filename="models/trained_fitness_model.pkl"):
        try:
            with open(filename, 'rb') as f:
                model_state = pickle.load(f)
            self.weights = model_state['weights']
            self.biases = model_state['biases']
            self.mean_X = model_state['mean_X']
            self.std_X = model_state['std_X']
            print(f"Модель завантажено з {filename}")
        except FileNotFoundError:
            print(f"Файл {filename} не знайдено — використовуємо ненавчену модель")
        except Exception as e:
            print(f"Помилка завантаження: {e}")