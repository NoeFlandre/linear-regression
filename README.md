# PyTorch Linear Regression Playground

A minimal PyTorch implementation of simple linear regression for educational purposes.

## 📂 Project Structure
- **`main.py`**: Training loop, data generation, and model execution.
- **`model.py`**: Definition of the `LinearRegressionModel` using `torch.nn`.

## 🚀 Quick Start

1. **Install Dependencies**
   Ensure you have PyTorch installed:
```
pip install torch
```

2. **Run the Training Loop**
Execute the main script to train the model on synthetic data:
```
python main.py
```


## ℹ️ Model Details
- **Architecture**: Single layer linear model (`nn.Linear`)
- **Input/Output**: 1D input features $\to$ 1D target
- **Optimization**: Stochastic Gradient Descent (SGD) with MSE Loss
- **Parameters**:
- Learning Rate: `0.01`
- Epochs: `1000`
- True Function: $y = 3x + 2 + \text{noise}$