# vpg-cartpole
The Vanilla Policy Gradeint Algorithm implemented for the OpenAI Gym's CartPole environment.

**My Hyperparameters:**
`Number of trajectories = 32
Learning rate for policy parameters: 0.1
Gamma: 0.99
Policy network architecture: (FC-layer(input_shape, 128), FC-layer(128, n_actions))
Advantage function: Discounted sum of rewards, no baseline.`



**The Rewards (Y-Axis) vs Episodes plot (X-Axis):**
![PLOT](/myplot.png)
