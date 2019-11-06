" Model file of Read,Watch, and Move Reinforcement Learning for Temporally Grounding Natural Language Descriptions in video\
 (https://arxiv.org/abs/1901.06829) "

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out))
    return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, mean=0, std=0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data)
        m.bias.data.fill_(0)


class A2C(nn.Module):
    def __init__(self):
        super(A2C, self).__init__()
        self.word_embedding_size = 300
        self.visual_feature_dim = 4096
        self.vocab_size = 1089

        self.gru1 = nn.GRU(input_size=self.word_embedding_size,
                             hidden_size=512, batch_first=True)
        self.gru2 = nn.GRU(input_size=512+self.visual_feature_dim,
                             hidden_size=512, batch_first=True)
        self.hidden2idx = nn.Linear(512, self.vocab_size)
        self.gobal_fc = nn.Linear(self.visual_feature_dim, 512)
        self.local_fc = nn.Linear(self.visual_feature_dim, 512)
        self.location_fc = nn.Linear(2, 128)
        self.state_fc = nn.Linear(512+512+512+512+128, 1024)

        self.gru = nn.GRUCell(1024, 1024)
        self.critic_linear = nn.Linear(1024, 1)
        self.actor_linear = nn.Linear(1024, 7) #7 action

        self.tiou_resfc = nn.Linear(1024, 1)
        self.location_resfc = nn.Linear(1024, 2)

        # Initializing weights
        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

    def forward(self, global_feature, local_feature, token_embeddings, target, location_feature, hidden_state):
        global_feature = self.gobal_fc(global_feature)
        global_feature_norm = F.normalize(global_feature, p=2, dim=1)
        global_feature_norm = F.relu(global_feature_norm)

        local_feature_norm = self.local_fc(local_feature)
        local_feature_norm = F.normalize(local_feature_norm, p=2, dim=1)
        local_feature_norm = F.relu(local_feature_norm)

        output, sentence_embedding = self.gru1(token_embeddings)
        sentence_embedding = sentence_embedding.view(-1,sentence_embedding.shape[-1])
        gru2_input = local_feature.unsqueeze(1).repeat(1, output.shape[1], 1)
        gru2_input = torch.cat((output, gru2_input), dim=2)
        output, self.hidden = self.gru2(gru2_input)
        output = self.hidden2idx(output)
        output = F.softmax(output, dim=2)
        reconstruction_prob = (output * target).sum(dim=2).mean()

        senetence_feature = self.hidden.squeeze(0)
        senetence_feature_norm = F.normalize(senetence_feature, p=2, dim=1)
        senetence_feature_norm = F.relu(senetence_feature_norm)

        location_feature = self.location_fc(location_feature)
        location_feature_norm = F.normalize(location_feature, p=2, dim=1)
        location_feature_norm = F.relu(location_feature_norm)

        state_feature = torch.cat([global_feature_norm, local_feature_norm, sentence_embedding, senetence_feature_norm, location_feature_norm], 1)
        state_feature = self.state_fc(state_feature)
        state_feature = F.relu(state_feature)

        hidden_state = self.gru(state_feature, hidden_state)
        value = self.critic_linear(hidden_state)
        actions = self.actor_linear(hidden_state)

        tIoU = self.tiou_resfc(state_feature)
        location = self.location_resfc(state_feature)

        return hidden_state, actions, value, tIoU, location, reconstruction_prob



