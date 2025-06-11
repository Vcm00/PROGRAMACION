# PROGRAMACION

This repository provides examples for building a simple multilayer perceptron (MLP) regression model with PyTorch. You can run the code either as a Jupyter notebook or directly from a Python script.

## Contents

- `mlp_regression.ipynb` – step-by-step notebook that loads data, trains a small MLP, and visualizes the results.
- `mlp_regression.py` – equivalent Python script you can execute from the command line.
- `mnist_mlp.py` – trains a simple MLP on the MNIST digits dataset and searches
  for good hyperparameters using Optuna.

## Running

Install the dependencies and execute the notebook or script:

```bash
pip install torch matplotlib scikit-learn pandas

# run as script
python mlp_regression.py
# train MNIST model and search hyperparameters
python mnist_mlp.py
```

[Open the notebook in Colab](https://colab.research.google.com/github/your-user/your-repo/blob/main/mlp_regression.ipynb)

## License

This project is released under the [MIT License](LICENSE).

