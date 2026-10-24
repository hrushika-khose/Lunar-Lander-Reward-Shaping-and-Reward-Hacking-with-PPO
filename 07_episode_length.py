import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

def avg_length(model_name, episodes=10):
    model = PPO.load(model_name)
    env = gym.make("LunarLander-v3")   # plain env, no wrapper
    lengths, scores = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, steps, total = False, 0, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(int(action))
            steps += 1
            total += r
            done = term or trunc
        lengths.append(steps)
        scores.append(total)
    print(f"{model_name}: avg frames = {np.mean(lengths):.0f}, avg score = {np.mean(scores):.1f}")

avg_length("ppo_lander_baseline")
avg_length("ppo_lander_penalty")
avg_length("ppo_lander_hack")
avg_length("ppo_lander_fixed")