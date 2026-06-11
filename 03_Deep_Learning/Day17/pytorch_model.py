import torch

model = torch.nn.Sequential(
    torch.nn.Linear(1, 4),
    torch.nn.ReLU(),
    torch.nn.Linear(4, 1),
    torch.nn.Sigmoid()
)

print(model)

x = torch.tensor([[5.0]])

prediction = model(x)

print(prediction)