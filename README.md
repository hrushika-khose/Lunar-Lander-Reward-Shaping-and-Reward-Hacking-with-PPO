# Lunar Lander: Reward Shaping and Reward Hacking with PPO

I trained an AI to land a rocket using reinforcement learning, then wrote my own reward rules, **deliberately broke one**, watched the AI exploit it, and fixed it.

The AI does not learn what you *mean*. It learns exactly what you *score*. This project shows that in practice.

## Demo

| Baseline (built-in reward) | Hover bonus (broken rule) | Fixed (improvement bonus) |
|---|---|
| Lands in ~365 frames | Hovers until the time limit | Lands in ~324 frames |



## What this project does

1. **Baseline:** train PPO on Lunar Lander with the game's built-in reward.
2. **Time penalty:** add a small per-frame fee (`TimePenalty`) to encourage faster landings.
3. **Hack:** add a flawed rule, +1 per frame while the rocket is upright and slow (`HoverBonus`).
4. **Fix:** pay for *improvement* instead of *sitting in a good state*, plus a small per-frame fee (`ImprovementBonus`).

## Results

Each model: PPO, 1,000,000 training steps, seed 0. Evaluated on the **plain environment (no wrapper)** over 10 episodes, so my own bonuses can't inflate the score. An episode is cut off at 1,000 frames, and 200+ average score counts as solved.

| Model | Avg frames | Avg score | Outcome |
|---|---|---|---|
| Baseline | 365 | 238.4 | Solved |
| Time penalty | 317 | 252.0 | Solved, slightly faster |
| Hover bonus (broken) | 987 | 14.6 | Hovers until time limit, almost never lands |
| Improvement bonus (fixed) | 324 | 253.1 | Solved, back to normal speed |

## What I learned

**Why the hover bonus breaks the AI.** An episode lasts up to 1,000 frames. Hovering can collect up to about +1,000 from the bonus while fuel costs at most about 300. Landing pays +100 and *ends the episode*, which stops the +1 payments. So hovering beats landing, and PPO learned exactly that. This is **reward hacking**.

**Why the fix works.** The `ImprovementBonus` pays only for the *change* in how calm the rocket is between frames (+1 for becoming calm, -1 for losing calm). Hovering earns 0 from the bonus and pays the per-frame fee, so waiting costs something and landing is the best choice again.

## Limitations (being honest)

- **One seed per model, 10 evaluation episodes.** Small differences between baseline, penalty and fixed are within noise. Only the hack result is a large, clear gap. Results vary between evaluation runs because each episode starts from a random state.
- **The fixed model changes two things at once** (improvement bonus *and* a 0.05 fee), so I can't say the bonus alone caused the recovery. Testing `ImprovementBonus` with `fee=0` would isolate it.
- Not tested across multiple seeds yet.

## Project structure

```
lunar-lander/
├── check.py                 # verify installs
├── 01_random.py             # random agent crashing
├── 02_train.py              # baseline PPO training
├── 03_evaluate.py           # evaluate a saved model
├── 04_record.py             # record baseline videos
├── 04_record_hack.py        # record hack videos
├── 04_record_fixed.py       # record fixed videos
├── 05_train_penalty.py      # train with TimePenalty
├── 06_train_hack.py         # train with HoverBonus (broken)
├── 07_episode_length.py     # compare frames and scores across models
├── 08_train_fixed.py        # train with ImprovementBonus (fixed)
└── rewards.py               # TimePenalty, HoverBonus, ImprovementBonus wrappers
```

## How to run

Requires Python 3.10+.

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install swig
pip install "gymnasium[box2d]" stable-baselines3 moviepy
```

Then run the scripts in order:

```bash
python check.py
python 01_random.py          # watch a random rocket crash
python 02_train.py           # baseline (about 20-40 min on a laptop, no GPU needed)
python 03_evaluate.py
python 04_record.py
python 05_train_penalty.py
python 06_train_hack.py
python 08_train_fixed.py
python 07_episode_length.py  # compare all models
```

## Tech stack

- Python
- [Gymnasium](https://gymnasium.farama.org/) (LunarLander-v3, Box2D physics)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (PPO, MlpPolicy)
- PyTorch (installed with Stable-Baselines3)

## Troubleshooting

- **Box2D install fails mentioning "swig":** run `pip install swig`, then reinstall `"gymnasium[box2d]"`.
- **`LunarLander-v2` not found:** current Gymnasium only has `LunarLander-v3`.
- **Crash on Windows during training:** keep training code inside `if __name__ == "__main__":`.
## Troubleshooting

- **Box2D install fails mentioning "swig":** run `pip install swig`, then reinstall `"gymnasium[box2d]"`.
- **`LunarLander-v2` not found:** current Gymnasium only has `LunarLander-v3`.
- **Crash on Windows during training:** keep training code inside `if __name__ == "__main__":`.
