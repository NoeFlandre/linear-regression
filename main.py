import torch
from torch import nn
from model import LinearRegressionModel

m = 3
b = 2
learning_rate = 0.01
epochs = 1000


X = torch.rand(100, 1)
Y = m*X + b + torch.randn(100, 1)

model = LinearRegressionModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    y_pred = model(X)
    train_loss = loss_fn(y_pred, Y)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch : {epoch} | Loss : {train_loss.item():.4f}")

