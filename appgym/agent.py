import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

class PolicyModel(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyModel, self).__init__()

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.ouput_layer = nn.Linear(hidden_size, output_size, bias=False)

        self.output_size = output_size

    def forward(self, x):
        x = x.view(1, -1)
        x = F.tanh(x) # Squash inputs
        x = F.relu(self.inp(x))
        x = self.out(x)

        scores = x[:, :self.output_size]
        value = x[:, self.output_size]
        return scores, value


class Agent:
    
    def __init__(self):
        self.model = PolicyModel()

    def act(self, state):
        state = Variable(torch.from_numpy(state).float())
        scores, value = policy(state) # Forward state through network
        scores = F.dropout(scores, drop, True) # Dropout for exploration
        scores = F.softmax(scores)
        action = scores.multinomial() # Sample an action

        return action, value
