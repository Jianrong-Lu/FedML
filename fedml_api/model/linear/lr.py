import torch


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        # outputs = torch.relu(x)
        # outputs = x
        outputs = torch.sigmoid(x)
        return x,outputs

class moon_LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(moon_LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        outputs = torch.sigmoid(x)
        return x,outputs
