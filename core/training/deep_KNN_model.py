from torch import nn

class KNNNet(nn.Module):
    """
    Uses basic dense layers and ReLU with a softmax activation.
    Batch norms help improve performance.
    """
    def __init__(self):
        super(KNNNet, self).__init__()
        self.n1 = 256
        self.n2 = 128
        self.n3 = 64
        self.num_features = 26
        self.fc1 = nn.Linear(self.num_features, self.n1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.n1, self.n2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(self.n2, self.n3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(self.n3, 10)
        self.softmax = nn.Softmax(dim=1)
        self.bn1 = nn.BatchNorm1d(self.n1)
        self.bn2 = nn.BatchNorm1d(self.n2)
        self.bn3 = nn.BatchNorm1d(self.n3)


    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.bn1(x)
        x = self.relu2(self.fc2(x))
        x = self.bn2(x)
        x = self.relu3(self.fc3(x))
        x = self.bn3(x)
        x = x.view(-1,64)
        x = self.fc4(x)
        x = self.softmax(x)
        return x