import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from collections import namedtuple

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
dtype = torch.FloatTensor


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# class PolicyModel(nn.Module):

    # def __init__(self, input_size, hidden_size, output_size):
        # super(PolicyModel, self).__init__()

        # self.input_layer = nn.Linear(input_size, hidden_size)
        # self.ouput_layer = nn.Linear(hidden_size, output_size, bias=False)

        # self.output_size = output_size

    # def forward(self, x):
        # x = x.view(1, -1)
        # x = F.tanh(x) # Squash inputs
        # x = F.relu(self.inp(x))
        # x = self.out(x)

        # scores = x[:, :self.output_size]
        # value = x[:, self.output_size]
        # return scores, value

class Model(nn.Module):

    def __init__(self, n_actions):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(320, n_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class Agent:
    
    def __init__(self, actions):
        self.model = Model(len(actions))
        self.model.type(dtype)
        self.memory = ReplayMemory(10000)
        self.optimizer = optim.RMSprop(self.model.parameters())
        self.actions = actions

    def act(self, state):
        state = Variable(torch.from_numpy(state).float())
        scores, value = policy(state) # Forward state through network
        scores = F.dropout(scores, drop, True) # Dropout for exploration
        scores = F.softmax(scores)
        action = scores.multinomial() # Sample an action

        return action, value

        # steps_done = 0
        # def select_action(state, actions):
            # sample = random.random()
            # eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
            # steps_done += 1
            # if sample > eps_threshold:
                # vals = model(Variable(state.type(dtype), volatile=True)).data[0]
                # max_idx = vals[:len(actions)].max(0)[1][0]
                # return torch.LongTensor([[max_idx]])
            # else:
                # return torch.LongTensor([[random.randrange(len(actions))]])

        # def optimize_model():
            # if len(memory) < BATCH_SIZE:
                # return
            # transitions = memory.sample(BATCH_SIZE)
            # # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
            # batch = Transition(*zip(*transitions))

            # # Compute a mask of non-final states and concatenate the batch elements
            # non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
            # # We don't want to backprop through the expected action values and volatile will save us
            # # on temporarily changing the model parameters' requires_grad to False!
            # non_final_next_states_t = torch.cat(tuple(s for s in batch.next_state if s is not None)).type(dtype)
            # non_final_next_states = Variable(non_final_next_states_t, volatile=True)
            # state_batch = Variable(torch.cat(batch.state))
            # action_batch = Variable(torch.cat(batch.action))
            # reward_batch = Variable(torch.cat(batch.reward))

            # # if USE_CUDA:
                # # state_batch = state_batch.cuda()
                # # action_batch = action_batch.cuda()

            # # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
            # state_action_values = model(state_batch).gather(1, action_batch).cpu()

            # # Compute V(s_{t+1}) for all next states.
            # next_state_values = Variable(torch.zeros(BATCH_SIZE))
            # next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0].cpu()
            # # Now, we don't want to mess up the loss with a volatile flag, so let's clear it.
            # # After this, we'll just end up with a Variable that has requires_grad=False
            # next_state_values.volatile = False
            # # Compute the expected Q values
            # expected_state_action_values = (next_state_values * GAMMA) + reward_batch

            # # Compute Huber loss
            # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

            # # Optimize the model
            # optimizer.zero_grad()
            # loss.backward()
            # for param in model.parameters():
                # param.grad.data.clamp_(-1, 1)
            # optimizer.step()




