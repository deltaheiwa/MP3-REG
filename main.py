import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from nn import SimpleNeuralNet


def load_and_split_data(filepath, test_ratio=0.2):
    try:
        data = np.loadtxt(filepath)
        X = data[:, 0:1]
        Y = data[:, 1:2]
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None, None, None, None, None, None
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None, None, None, None, None

    indices = np.arange(len(data))
    np.random.shuffle(indices)

    split_idx = int(len(data) * (1 - test_ratio))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    X_train, Y_train = X[train_indices], Y[train_indices]
    X_test, Y_test = X[test_indices], Y[test_indices]

    return X_train, Y_train, X_test, Y_test, X, Y


if __name__ == "__main__":
    hidden_size_to_use = 15

    dane = Path(os.getcwd()).joinpath('Dane').resolve()
    file_list = list(dane.glob('*.txt'))
    file_list.sort()

    results = {}

    print("--- Processing all files for MSE results ---")
    if not file_list:
        print("No 'dane*.txt' files found in the current directory.")
    else:
        for filename in file_list:
            X_train, Y_train, X_test, Y_test, X_full, Y_full = load_and_split_data(filename)

            if X_train is None:
                continue

            current_results = {}

            nn_batch_tanh = SimpleNeuralNet(hidden_size=hidden_size_to_use, activation='tanh')
            nn_batch_tanh.train(X_train, Y_train, epochs=2000, learning_rate=0.05)
            train_mse_bt = np.mean(np.square(nn_batch_tanh.forward(X_train) - Y_train))
            test_mse_bt = np.mean(np.square(nn_batch_tanh.forward(X_test) - Y_test))
            current_results["Batch_Tanh"] = {"Train MSE": train_mse_bt, "Test MSE": test_mse_bt, "model": nn_batch_tanh}

            nn_batch_relu = SimpleNeuralNet(hidden_size=hidden_size_to_use, activation='relu')
            nn_batch_relu.train(X_train, Y_train, epochs=2000, learning_rate=0.01)
            train_mse_br = np.mean(np.square(nn_batch_relu.forward(X_train) - Y_train))
            test_mse_br = np.mean(np.square(nn_batch_relu.forward(X_test) - Y_test))
            current_results["Batch_ReLU"] = {"Train MSE": train_mse_br, "Test MSE": test_mse_br, "model": nn_batch_relu}

            results[filename] = current_results
            print(f"Results for {filename}:")
            print(f"  Batch+Tanh:  Train MSE: {train_mse_bt:.4f}, Test MSE: {test_mse_bt:.4f}")
            print(f"  Batch+ReLU:  Train MSE: {train_mse_br:.4f}, Test MSE: {test_mse_br:.4f}")
            print("-" * 30)

        if file_list and results:
            first_file = file_list[0]
            first_file_results = results[first_file]

            _, _, _, _, X_full_viz, _ = load_and_split_data(first_file)
            X_train_viz, Y_train_viz, X_test_viz, Y_test_viz, _, _ = load_and_split_data(first_file)

            print(f"\n--- Generating detailed visualization example for {first_file} ---")

            min_test_mse = float('inf')
            best_model_name = ""
            best_model = None
            for name, m_results in first_file_results.items():
                if m_results["Test MSE"] < min_test_mse:
                    min_test_mse = m_results["Test MSE"]
                    best_model_name = name
                    best_model = m_results["model"]

            print(f"Plotting best performing model: {best_model_name} (Test MSE: {min_test_mse:.4f})")

            X_min, X_max = X_full_viz.min(), X_full_viz.max()
            X_plot_viz = np.linspace(X_min, X_max, 100).reshape(-1, 1)

            Y_plot_pred_viz = best_model.forward(X_plot_viz)

            plt.figure(figsize=(10, 6))

            plt.scatter(X_train_viz, Y_train_viz, color='blue', label='Training Data', alpha=0.6, s=15)
            plt.scatter(X_test_viz, Y_test_viz, color='green', label='Testing Data', alpha=0.6, s=15, marker='x')

            plt.plot(X_plot_viz, Y_plot_pred_viz, color='red', label=f'Model Fit ({best_model_name})', linewidth=2.5)

            plt.title(f'Neural Network Approximation - Dataset: {first_file}')
            plt.xlabel('X (Input)')
            plt.ylabel('Y (Target)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)

            plt.show()