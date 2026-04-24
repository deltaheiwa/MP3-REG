import numpy as np

class SimpleNeuralNet:
    def __init__(self, input_size=1, hidden_size=10, output_size=1, activation='tanh'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_name = activation

        if activation == 'relu':
            self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        else:
            self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1. / input_size)

        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def _activate(self, Z):
        if self.activation_name == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        elif self.activation_name == 'tanh':
            return np.tanh(Z)
        elif self.activation_name == 'relu':
            return np.maximum(0, Z)

    def _activation_derivative(self, A, Z):
        if self.activation_name == 'sigmoid':
            return A * (1 - A)
        elif self.activation_name == 'tanh':
            return 1 - np.power(A, 2)
        elif self.activation_name == 'relu':
            return np.where(Z > 0, 1, 0)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self._activate(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.Z2
        return self.A2

    def backward(self, X, Y):
        m = X.shape[0]

        dZ2 = self.A2 - Y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self._activation_derivative(self.A1, self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        return dW1, db1, dW2, db2

    def update_params(self, dW1, db1, dW2, db2, learning_rate):
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X_train, Y_train, epochs, learning_rate):
        mse_history = []
        m = X_train.shape[0]

        for epoch in range(epochs):
            predictions = self.forward(X_train)
            dW1, db1, dW2, db2 = self.backward(X_train, Y_train)
            self.update_params(dW1, db1, dW2, db2, learning_rate)

            if epoch % 100 == 0:
                loss = np.mean(np.square(self.forward(X_train) - Y_train))
                mse_history.append(loss)

        return mse_history
