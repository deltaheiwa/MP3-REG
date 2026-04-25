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

            nn_online_tanh = SimpleNeuralNet(hidden_size=hidden_size_to_use, activation='tanh')
            nn_online_tanh.train(X_train, Y_train, epochs=2000, learning_rate=0.01, method='online')
            train_mse_ot = np.mean(np.square(nn_online_tanh.forward(X_train) - Y_train))
            test_mse_ot = np.mean(np.square(nn_online_tanh.forward(X_test) - Y_test))
            current_results["Online_Tanh"] = {"Train MSE": train_mse_ot, "Test MSE": test_mse_ot, "model": nn_online_tanh}

            nn_online_relu = SimpleNeuralNet(hidden_size=hidden_size_to_use, activation='relu')
            nn_online_relu.train(X_train, Y_train, epochs=2000, learning_rate=0.01)
            train_mse_or = np.mean(np.square(nn_online_relu.forward(X_train) - Y_train))
            test_mse_or = np.mean(np.square(nn_online_relu.forward(X_test) - Y_test))
            current_results["Online_ReLU"] = {"Train MSE": train_mse_or, "Test MSE": test_mse_or, "model": nn_online_relu}

            results[filename] = {
                "data": (X_train, Y_train, X_test, Y_test, X_full),
                "models": current_results
            }

            print(f"Results for {filename}:")
            print(f"  Batch+Tanh:  Train MSE: {train_mse_bt:.4f}, Test MSE: {test_mse_bt:.4f}")
            print(f"  Batch+ReLU:  Train MSE: {train_mse_br:.4f}, Test MSE: {test_mse_br:.4f}")
            print(f"  Online+Tanh: Train MSE: {train_mse_ot:.4f}, Test MSE: {test_mse_ot:.4f}")
            print(f"  Online+ReLU: Train MSE: {train_mse_or:.4f}, Test MSE: {test_mse_or:.4f}")
            print("-" * 30)

        if file_list and results:
            print("\n--- Beginning Sequential Visualization ---")
            print("Close each plot window to view the next one.\n")

            for filepath in file_list:
                X_train, Y_train, X_test, Y_test, X_full = results[filepath]["data"]
                file_models = results[filepath]["models"]

                X_min, X_max = X_full.min(), X_full.max()
                X_plot_viz = np.linspace(X_min, X_max, 100).reshape(-1, 1)

                for model_name, m_results in file_models.items():
                    print(f"Plotting {filepath.name} -> {model_name}")

                    model = m_results["model"]
                    Y_plot_pred_viz = model.forward(X_plot_viz)

                    plt.figure(figsize=(10, 6))

                    plt.scatter(X_train, Y_train, color='blue', label='Training Data', alpha=0.6, s=15)
                    plt.scatter(X_test, Y_test, color='green', label='Testing Data', alpha=0.6, s=15, marker='x')

                    plt.plot(X_plot_viz, Y_plot_pred_viz, color='red', label=f'Model Fit ({model_name})', linewidth=2.5)

                    plt.title(f'Dataset: {filepath.name} | Model: {model_name}')
                    plt.xlabel('X (Input)')
                    plt.ylabel('Y (Target)')

                    test_mse = m_results['Test MSE']
                    plt.plot([], [], ' ', label=f'Test MSE: {test_mse:.4f}')

                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.7)

                    plt.show()
