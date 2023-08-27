import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class VectorReconstruction(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
                nn.Linear(1,16),
                nn.ReLU(),
                nn.Linear(16,1),)

    def forward(self, x):
        x = self.model(x)
        return torch.atan2(torch.sine(x), torch.cosine(x))

# Hyperparameters
epochs = 100

def vonMisesLoss(output, target):
    return torch.sum(1 - torch.exp(torch.cos(output - target))) 

     model.train()
#     for epoch in range(epochs):
#         data = data.to(device)
#         target = target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()

if __name__ == "__main__":
     pass
