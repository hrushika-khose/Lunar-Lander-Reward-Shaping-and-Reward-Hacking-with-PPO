import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("LunarLander-v3")
model = PPO.load("ppo_lander_baseline")

mean, std = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Average score: {mean:.1f} ± {std:.1f}")
print("SOLVED!" if mean >= 200 else "Not solved yet - train longer or try another seed")