import torch

data = torch.tensor([1, 2, 3])
gradient = torch.tensor(
    2.0,
    requires_grad=True
)

y = gradient ** 2 # forward pass
y.backward() # backward pass to compute the gradient of `y` with respect to `gradient`

print(gradient.grad)  # This will print the gradient of `y` with respect to `data`
print(y)   