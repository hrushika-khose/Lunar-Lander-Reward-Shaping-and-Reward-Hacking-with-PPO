from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

if __name__ == "__main__":
    env = make_vec_env("LunarLander-v3", n_envs=16)

    model = PPO(
        "MlpPolicy",        # standard small neural network
        env,
        n_steps=1024,       # frames collected per game before each learning round
        batch_size=64,
        n_epochs=4,
        gamma=0.999,        # how much it cares about future points
        gae_lambda=0.98,
        ent_coef=0.01,      # a little randomness so it keeps exploring
        verbose=1,
        seed=0,
    )

    model.learn(total_timesteps=1_000_000)   # 1 step = 1 frame
    model.save("ppo_lander_baseline")
    print("Saved!")