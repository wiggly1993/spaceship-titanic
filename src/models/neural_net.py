import torch
import torch.nn as nn


class SpaceshipNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(p=0.2)

        self.hidden1 = nn.Linear(25, 50)  
        self.hidden2 = nn.Linear(50, 150)
        self.hidden3 = nn.Linear(150, 300)
        self.hidden4 = nn.Linear(300, 150)
        self.hidden5 = nn.Linear(150, 50) 
        self.output = nn.Linear(50, 1)  

        # Define activation function, you can use others like ReLU or LeakyReLU
        self.activation = nn.ReLU()

    def forward(self, x):
        # Forward pass through each layer
        x = self.activation(self.hidden1(x))
        x = self.drop(x)
        x = self.activation(self.hidden2(x))
        x = self.drop(x)
        x = self.activation(self.hidden3(x))
        x = self.drop(x)
        x = self.activation(self.hidden4(x))
        x = self.drop(x)
        x = self.activation(self.hidden5(x))
        x = self.drop(x)
        x = self.output(x)
        return x


if __name__ == "__main__":
    dummy_passenger = torch.randn(1, 25)
    # Create an instance of the network
    net = SpaceshipNet()

    # Print the network structure
    out = net(dummy_passenger)

    print(out)