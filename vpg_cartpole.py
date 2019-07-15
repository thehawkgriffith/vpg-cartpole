import torch
import torch.nn as nn
import numpy as np
import gym
import matplotlib.pyplot as plt


class PolicyNet(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax()
        )

    def forward(self, x):
        x = x.view(1, -1)
        return self.fc(x)


class Baseline(nn.Module):

    def __init__(self, input_shape):
        super(Baseline, self).__init__()
        shape = 1
        for dim in input_shape:
            shape *= dim
        self.fc = nn.Sequential(
            nn.Linear(shape, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = x.view(1, -1)
        return self.fc(x)


class Agent:

    def __init__(self, policy_net, baseline_net):
        self.policy_net = policy_net
        self.baseline = baseline_net

    def train(self, env, num_traj, iterations, gamma, base_epochs):
        iter_rewards = []
        for iter in range(iterations):
            trajectories = []
            ITER_REW = 0
            for _ in range(num_traj):
                rewards = []
                log_probs = []
                s = env.reset()
                done = False
                while not done:
                    s = torch.FloatTensor([s]).cuda()
                    a = self.policy_net(s)
                    del s
                    a2 = a.detach().cpu().numpy()
                    vec = [0, 1]
                    u = np.random.choice(vec, 1, replace=False, p=a2[0])
                    log_probs.append(a[0][u])
                    del a
                    sp, r, done, _ = env.step(u[0])
                    if done:
                        if len(rewards) < 50:
                            r = -200
                    ITER_REW += r
                    rewards.append(r)
                    # env.render()
                    s = sp
                trajectories.append({'log_probs': log_probs, 'rewards': rewards})
            # self.update_baseline(base_epochs, trajectories, gamma)
            self.update_policy(trajectories, gamma)
            print("ITERATION:", iter+1, "AVG REWARD:", ITER_REW/num_traj)
            iter_rewards.append(ITER_REW/num_traj)
        return iter_rewards

    def update_baseline(self, epochs, trajectories, gamma):
        criterion = torch.nn.MSELoss()
        optim = torch.optim.Adam(self.baseline.parameters())
        for epoch in range(epochs):
            loss = torch.tensor(0).float().cuda()
            for trajectory in trajectories:
                for t in range(len(trajectory)):
                    r_t = 0
                    for t_d in range(t, len(trajectory)):
                        r_t += gamma**(t_d - t) * trajectory[t_d]['r']
                    pred = self.baseline(trajectory[t]['s'])
                    loss += criterion(pred, torch.FloatTensor([r_t]).cuda())
            print(loss.item())
            loss.backward()
            optim.step()
            optim.zero_grad()

    def update_policy(self, trajectories, gamma):
        loss = torch.tensor([0]).float().cuda()
        optim = torch.optim.Adam(self.policy_net.parameters(), lr=0.1)
        for trajectory in trajectories:
            for t in range(len(trajectory['rewards'])):
                r_t = 0
                log_prob = trajectory['log_probs'][t]
                temp = trajectory['rewards'][t:]
                for i, reward in enumerate(temp):
                    r_t += gamma**i * reward
                # for t_d in range(t, len(trajectory)):
                #     r_t += gamma ** (t_d - t) * trajectory['rewards'][t_d]
                # advantage += torch.FloatTensor([r_t]).cuda() - self.baseline(trajectory[t]['s'])[0]
                advantage = torch.FloatTensor([r_t]).cuda()
                loss += -log_prob * advantage
                # loss.backward()
            # loss += -log_probs * advantage
        loss = loss/len(trajectories)
        loss.backward()
        # print("\nBefore zerograd\n")
        # for name, param in self.policy_net.named_parameters():
        #     print(name, param.grad.data.sum())
        optim.step()
        optim.zero_grad()
        # print("\nAfter zerograd\n")
        # for name, param in self.policy_net.named_parameters():
        #     print(name, param.grad.data.sum())


def main():
    env = gym.make('CartPole-v0')
    policy_net = PolicyNet(env.observation_space.shape, env.action_space.n).to(torch.device('cuda'))
    base_net = Baseline(env.observation_space.shape).to(torch.device('cuda'))
    # policy_net.load_state_dict(torch.load('./policynet'))
    # base_net.load_state_dict(torch.load('./basenet'))
    agent = Agent(policy_net, base_net)
    rews = agent.train(env, 32, 200, 0.99, 5)
    plt.plot(rews)
    plt.show()


main()