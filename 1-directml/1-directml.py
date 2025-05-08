import time
import torch
import torch.nn as nn
import torch_directml

device = torch_directml.device()
tensor1 = torch.tensor([1]).to(device)
tensor2 = torch.tensor([2]).to(device)
result = tensor1 + tensor2
print(result.item())

print(torch.__version__)

print(device)

start = time.time()

x = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0], [4.0]], dtype=torch.float).to(device)
y = torch.tensor([[-7.0], [-3.0], [-1.0], [1.0], [3.0], [5.0], [7.0]], dtype=torch.float).to(device)

## Neural network with 1 hidden layer
layer1 = nn.Linear(1, 1, bias=False)
model = nn.Sequential(layer1).to(device)

## loss function
criterion = nn.MSELoss()

## optimizer algorithm
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

prev_loss = 0
same_loss_count = 0

## training
for ITER in range(150):
    model = model.train()

    ## forward
    output = model(x)
    loss = criterion(output, y)
    optimizer.zero_grad()

    ## backward + update model params
    loss.backward()
    loss_item = round(loss.detach().item(), 4)
    optimizer.step()

    model.eval()
    print('Epoch: %d | Loss: %.4f' % (ITER, loss_item))

    if loss_item == prev_loss:
        same_loss_count += 1
        if same_loss_count >= 3:
            break
    else:
        same_loss_count = 0
    prev_loss = loss_item

end = time.time()

print('\nTime: %.2fs\n' % (end - start))

## test the model
sample = torch.tensor([10.0], dtype=torch.float).to(device)
predicted = model(sample)
predicted_item = predicted.detach().item()
correct = 10 * 2 - 1
print(f"Predicted: {predicted_item}, Correct: {correct}, Difference: {predicted_item - correct}")
