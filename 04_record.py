import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

model = PPO.load("ppo_lander_baseline")

env = gym.make("LunarLander-v3", render_mode="rgb_array")  # draws to memory, not a window
env = RecordVideo(env, video_folder="videos_baseline",
                  episode_trigger=lambda ep: True,          # record every episode
                  name_prefix="baseline")

for ep in range(3):
    obs, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)  # best action, no randomness
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

env.close()
print("Videos saved in videos_baseline/")