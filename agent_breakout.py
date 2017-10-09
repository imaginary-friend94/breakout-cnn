from model import AgentNeuralNetwork
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from scipy.misc import imresize
import numpy as np
import time
import tqdm


class AgentBO():
    def __init__(self, n_actions, input_channel):
        self.model = AgentNeuralNetwork(input_channel, n_actions)
        self.model.cpu()
        self.n_actions = n_actions

    def get_action(self, X):
        X = np.transpose(X, (2, 0, 1))[np.newaxis, ...]
        X = Variable(torch.Tensor(X.astype(np.float64)))
        prob = self.model.forward(X).data.numpy()
        return np.random.choice(self.n_actions, p=prob.reshape(-1))

    def start_game_train(self, env, option, verbose = False):
        learning_rate = option['learning_rate']
        max_iter = option['max_iter']
        n_epoch = option['n_epoch']
        session_per_epoch = option['session_per_epoch']
        frame_pass = option['frame_pass']
        batch_size = option['batch_size']
        self.global_reward = np.array([]).astype(np.int8)

        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        for index_epoch in range(n_epoch):

            time_ = time.time()

            session_out = [self.start_session(env, max_iter, frame_pass) for _ in range(session_per_epoch)]
            batch_actions, batch_observ, batch_rewards = map(np.array, zip(*session_out))

            self.global_reward = np.concatenate([self.global_reward, batch_rewards])
            self.act = batch_actions

            print("session is generated!")

            threshold = np.percentile(batch_rewards, 75)

            print("threshold : %f" % threshold)

            best_actions = batch_actions[batch_rewards > threshold]
            best_observ = batch_observ[batch_rewards > threshold]

            del batch_observ, batch_actions, batch_rewards

            if (best_observ.shape[0] == 0):
                continue

            #print(best_observ[0].shape)

            best_observ, best_actions = map(np.concatenate, [best_observ, best_actions])

            print("observ dim : " + str(best_observ.shape))

            best_observ = np.transpose(best_observ, (0, 3, 1, 2))

            print("start train")

            loss = self.train_batch(batch_size, best_observ, best_actions, optimizer, criterion)

            if verbose:
                print("mean reward : %f; loss : %f; time : %f;" % (threshold,
                                                                    float(loss),
                                                                    time.time() - time_))

        return 0

    def train_batch(self, batch_size, best_observ, best_actions, optimizer, criterion):
        optimizer.zero_grad()
        array_size = best_actions.shape[0]
        batch_count = int(array_size / batch_size)
        loss_out = 0

        for index_batch in tqdm.tqdm(range(batch_count)):

            batch_observ = best_observ[index_batch : index_batch + batch_size]
            batch_actions = best_actions[index_batch: index_batch + batch_size]

            batch_observ= Variable(torch.FloatTensor(batch_observ.astype(np.float64)))
            batch_actions = Variable(torch.LongTensor(batch_actions.astype(np.int)))

            output = self.model(batch_observ)
            loss = criterion(output, batch_actions)
            loss_out += loss.data[0]
            loss.backward()
            optimizer.step()

        return loss_out / batch_count

    def start_session(self, env, max_iter, frame_pass):
        rs = lambda X : imresize(X, (150, 100), interp="nearest")
        actions_list = []
        observ_list = []
        observ_conc_list = []
        total_reward = 0
        observ_list.append(rs(env.reset()))
        observ_list.append(rs(env.step(env.action_space.sample())[0]))
        observ_list.append(rs(env.step(env.action_space.sample())[0]))
        for iter_game in range(max_iter):
            observ_conc = np.concatenate(observ_list[-3:], axis=1)
            observ_conc = observ_conc / 255.0
            action = self.get_action(observ_conc)
            assert env.action_space.contains(action)

            observ, reward, done, info = env.step(action)
            observ = rs(observ)
            total_reward += reward
            actions_list.append(action)
            observ_list.append(observ)
            observ_conc_list.append(observ_conc)

            if len(observ_list) > 4:
                observ_list.pop()

            if done:
                break

        return actions_list, observ_conc_list, total_reward

    def save_model(self, name):
        torch.save(self.model.state_dict(), name)

    def load_model(self, name):
        self.model.load_state_dict(torch.load(name))
