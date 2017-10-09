import torch.nn as nn

class AgentNeuralNetwork(nn.Module):

    def __init__(self, input_channel, n_actions):
        super(AgentNeuralNetwork, self).__init__()
        self.conv1, self.maxpool1, self.dropout1, self.activation1, self.batchnorm1 = \
            self.create_conv_block(input_channel, 14, 8)
        self.conv2, self.maxpool2, self.dropout2, self.activation2, self.batchnorm2 = \
            self.create_conv_block(14, 20, 8)
        self.conv3, self.maxpool3, self.dropout3, self.activation3, self.batchnorm3 = \
            self.create_conv_block(20, 25, 5)
        self.conv3, self.maxpool3, self.dropout3, self.activation3, self.batchnorm3 = \
            self.create_conv_block(25, 20, 5)
        self.conv3, self.maxpool3, self.dropout3, self.activation3, self.batchnorm3 = \
            self.create_conv_block(20, n_actions, 3)
        self.n_actions = n_actions
        self.layers = []

    def __call__(self, X):
        return self.forward(X)

    def create_conv_block(self, in_channel, out_channel, kernel_size, prob_dropout = 0.5):
        conv = nn.Conv2d(in_channel, out_channel, kernel_size, dilation=1)
        maxpool = nn.MaxPool2d(2)
        dropout = nn.Dropout2d(p = prob_dropout)
        activation = nn.ReLU()
        bn = nn.BatchNorm2d(out_channel)
        return conv, maxpool, dropout, activation, bn

    def forward_conv_block(self, X, layers):
        conv, maxpool, dropout, activation, batchnorm = layers
        X = conv(X)
        X = batchnorm(X)
        X = activation(X)
        X = dropout(X)
        X = maxpool(X)
        return X

    def global_max_pooling(self, X):
        return nn.MaxPool2d(X.size()[2:])(X)


    def forward(self, X):
        X = self.forward_conv_block(X, [self.conv1, self.maxpool1, self.dropout1, self.activation1, self.batchnorm1])
        X = self.forward_conv_block(X, [self.conv2, self.maxpool2, self.dropout2, self.activation2, self.batchnorm2])
        X = self.forward_conv_block(X, [self.conv3, self.maxpool3, self.dropout3, self.activation3, self.batchnorm3])
        X = self.global_max_pooling(X)
        return nn.Softmax()(X.view(X.size()[0], self.n_actions))




