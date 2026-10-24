import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")  # human = show the window
obs, info = env.reset(seed=0)

for frame in range(600):
    action = env.action_space.sample()   # random button: 0, 1, 2 
    obs, reward, terminated, truncated, info = env.step(action)

    if frame % 20 == 0:
        print(f"frame {frame}  action={action}  reward={reward:.2f}")
        print("  observation:", obs.round(2))

    if terminated or truncated:          # crashed, landed, or 
        print("Episode over. Restarting.")
        obs, info = env.reset()

env.close()