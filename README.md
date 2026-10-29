# DSAN-6600-Project

Semester Project for DSAN 6600: Deep Learning and Neural Networks

We use [ACRL](https://github.com/Jurredr/ACRL), a Soft Actor-Critic (SAC) implementation for Assetto Corsa, to train an agent that can drive a full, valid lap of a race track. If it gets there, our next target is the lap time set by Assetto Corsa's built-in AI on the same car and track.

**Status (Check-In 2):** Our SAC baseline trains end to end in Assetto Corsa. We have a five-episode pilot, one overnight baseline run (570 episodes), one run with a revised reward and early termination for stuck episodes (189 episodes), and a 61-episode continuation of that run resumed from its saved model. The baseline learned to stand still. The revised reward stopped that collapse, and resuming from a saved model now works, but the car still does not get further around the lap. This check-in documents the baseline, what it produced, why it fails, and what we will change for Check-In 3 to get the car around a full lap.

Items marked **(To be added for CI2: ...)** are still needed for this submission. Items marked (To be added: ...) are planned for later check-ins.

## Check-In 2 at a Glance

| Rubric item | Where to look | Status |
| --- | --- | --- |
| Neural baseline | [2. Neural Baseline](#2-neural-baseline-sac) | SAC runs end to end; training setup documented |
| Results + evaluation | [3. Results and Evaluation](#3-results-and-evaluation) | Training-run results and learning curves done; held-out evaluation |
| Failure analysis | [4. Failure Analysis](#4-failure-analysis) | Six failures with evidence, causes, and fixes (model loading now fixed); next steps toward CI3 |
| Repo / write-up | [1.3 Repository and How to Run](#13-repository-and-how-to-run) | Layout and run steps; final layout |

**Key result:** the baseline is still far from Stage 1 (completing a lap). Over 570 training episodes, the policy learned to move *less* (median speed fell from 3.4 to 0.8 km/h) while its reward improved slightly. The network learned what the reward asked for. The reward asked for the wrong thing. A revised reward kept the car moving (median speed held at 2–3 km/h instead of collapsing), but every episode still ended with the car stuck, and the farthest it got (2.4% of a lap) is no better than before. Resuming that run from its saved model for another 61 episodes brought it back to the same level, not beyond it.

## 1. Project Overview

### 1.1 Goal

AI opponents largely decide how challenging and fun a racing simulator feels. Unlike chess or Go, racing needs continuous control of steering, throttle, and brake while reacting to how the car moves, and one small mistake can ruin a corner or invalidate a lap. We want to see how far an agent that learns only by driving can get.

Our goals come in two stages:

| Stage | Goal | How we measure it |
| --- | --- | --- |
| **1. Complete a lap (main goal)** | The agent drives one full, valid lap of Silverstone 1967 without help. | Valid-lap completion rate in held-out evaluation; environment steps until the first valid lap |
| **2. Catch the built-in AI (stretch goal)** | Once the agent completes laps reliably, close the gap to Assetto Corsa's built-in AI in the same car on the same track. | Median valid lap time compared with the built-in AI's lap of **1:25.454** |

Everything we change before Stage 2 serves Stage 1. In practice that means adjusting the reward, analyzing why episodes fail, and fixing whatever stops the car from getting around the track. A faster lap only counts once the car can finish laps reliably, so lap times are always reported together with the completion rate.

With limited time and compute, we change one thing at a time and compare it with a reproducible baseline under matched conditions. Matching the built-in AI's lap time would mean the agent is competitive with the game's own opponents. It would not, by itself, mean the agent drives at a human expert's level.

How we plan to get to Stage 1:

| Approach | Hypothesis | Controlled comparison |
| --- | --- | --- |
| **A. Adjust the reward** | The baseline reward makes standing still the safest choice (see [4.1](#41-the-policy-learned-to-stand-still)). A reward that pays for forward progress should get the car moving and further around the lap. | Compare the baseline reward with the revised reward, keeping everything else the same. |
| **B. Test the observations** | Absolute coordinates may help on a single track, but we don't know how much they actually contribute. | Mask the absolute coordinates while keeping the network architecture and other inputs unchanged. (Lower priority until the car completes laps.) |
| **C. Analyze failures** | Repeated failures, such as getting stuck or running off at the same corner, waste training steps and keep the car from finishing. | Find where and why episodes fail, then test one targeted fix at a time. |

We will start with A, guided by C: the CI2 baseline's main failure comes from the reward.

| Decision | Current plan |
| --- | --- |
| Main goal | Complete one full, valid lap (Stage 1) |
| First intervention | Revised reward (A) |
| Stage 2 trigger | (To be added: the completion rate at which we move on to lap time, e.g. a share of held-out episodes) |

### 1.2 Scope

One car, one track, no opponents, fixed driving conditions.

| Setting | Value |
| --- | --- |
| Car and track | Ferrari 458 GT2 on Silverstone 1967 (the defaults in `main.py`) |
| Session settings | (To be added: tires, assists, weather, grip, fuel) |
| Hardware | Team member's Windows laptop |
| Training budget so far | 5,000 steps (pilot), 570,000 steps (overnight baseline, ~6.9 hours), 119,138 steps (revised reward, ~1.8 hours), and 29,900 steps (revised reward resumed, ~0.5 hours) |

### 1.3 Repository and How to Run

```
ACRL_modified_code/
├── ACRL.py                  # In-game app: sends telemetry from Assetto Corsa over a socket
└── standalone/
    ├── main.py              # Entry point: car/track settings, hyperparameters, start or resume training
    ├── ac_controller.py     # Sends actions to the game through a virtual Xbox controller (vgamepad)
    ├── ac_socket.py         # Socket connection to the in-game app
    ├── test_gamepad.py      # Checks that controller inputs reach the game (added by us)
    └── sac/
        ├── ac_environment.py   # Gymnasium environment: observations, reward, termination
        ├── sac.py              # SAC agent, replay buffer, training loop
        ├── core.py             # Actor and critic networks
        └── utils/
            ├── track_spline.py # Heading error and distance to the reference path
            └── logx.py         # Logging: progress.txt, config.json, model.pt, drive_data.json
track_data/spline_points.csv    # Reference path (from upstream ACRL)
experiments/<run name>/         # Created for each training run
```

**Running a training run**

1. Install Assetto Corsa and follow the upstream [setup instructions](https://github.com/Jurredr/ACRL#readme) (Visual C++ redistributable, MS-MPI, Python 3.3.5 for the in-game app).
2. Copy the in-game app into `assettocorsa/apps/python/ACRL` and enable it in the game settings.
3. Start a practice session with the chosen car and track.
4. Optional: run `python test_gamepad.py` from `standalone/` and switch to the game window to confirm throttle, steering, and brake reach the car.
5. Run `python main.py` from `standalone/`. It asks whether to load a previous model and for an experiment name, then trains. To resume, answer `y`, give the run folder (for example `experiments/reward_check_2`), and enter a **new** experiment name; reusing the old name overwrites that run's `progress.txt` and `model.pt`.
6. Results go to `experiments/<run name>/`: `progress.txt` (one row per episode), `config.json`, `model.pt`, and `drive_data.json`.

Resuming from a saved model works as of our latest code; see [4.2](#42-saved-models-could-not-be-loaded).

## 2. Neural Baseline (SAC)

### 2.1 Task as a Reinforcement Learning Problem

There is no fixed dataset. The agent collects its own training data by driving, and SAC learns from a replay buffer of that experience.

| Component | Definition |
| --- | --- |
| Observation (10 values) | Lap progress, speed (km/h), world x/y/z position, lap-invalid flag, lap count, previous lap progress, heading error relative to the reference path, distance from the reference path |
| Action (2 values, each in [-1, 1]) | Combined throttle/brake (positive = throttle, negative = brake) and steering |
| Reward (baseline) | `cos(θ) − \|sin(θ)\| − \|distance from reference path\| + speed / max_speed`, where θ is the heading error (ACRL's `_get_reward_5`) |
| Reward (revised run) | Our rewritten progress + speed reward (`reward_id=4`): scaled progress along the lap, a small speed term, a graded low-speed penalty, and a one-time off-track penalty, plus a penalty when an episode ends because the car is stuck (see [3.1](#31-training-runs)) |
| Episode end | Baseline: lap complete (progress ≥ 0.99 or lap count > 1), or 1,000 steps. Revised run: also ends when the car stays nearly stopped for 100 steps; limit raised to 4,000 steps |
| Step time | About 42 ms in our runs (not fixed; see [3.3](#33-log-artifacts)) |

### 2.2 Model

| Component | Configuration |
| --- | --- |
| Policy (actor) | MLP, 10 → 256 → 256 (ReLU), outputting the mean and log standard deviation of a Gaussian; actions squashed with tanh |
| Critics | Two Q-networks, each an MLP of 12 (state + action) → 256 → 256 (ReLU) → 1 |
| Target networks | Copies of both critics, updated by Polyak averaging |
| Parameters | About 208,000 in total (policy ~70,000; each critic ~69,000), computed from the layer sizes |

Source: [core.py](https://github.com/Jurredr/ACRL/blob/master/standalone/sac/core.py). We kept the network small on purpose: compute is limited, and the interventions we plan change the reward or the inputs, not the architecture.

### 2.3 Training Setup

**Loss functions** ([sac.py](https://github.com/Jurredr/ACRL/blob/master/standalone/sac/sac.py))

- **Critic loss:** mean squared error between each Q-network's estimate and the soft Bellman target  
  `y = r + γ (1 − done) [ min(Q1_target, Q2_target)(s', a') − α log π(a' | s') ]`, with `a'` sampled from the current policy. The two critic losses are summed.
- **Policy loss:** `mean( α log π(a | s) − min(Q1, Q2)(s, a) )`, which pushes the policy toward actions with high Q-value while keeping its entropy up.

**Optimizer and hyperparameters** (as set in `main.py`)

| Hyperparameter | Value |
| --- | --- |
| Optimizer | Adam, separate optimizers for the policy and the critics |
| Learning rate | 1e-3 (both) |
| Discount γ | 0.99 |
| Entropy coefficient α | 0.2 (fixed, not auto-tuned) |
| Polyak (target update) | 0.999 |
| Batch size | 32 |
| Replay buffer size | 1,000,000 |
| Random-action steps (`start_steps`) | 10,000 for a new run; 0 when resuming from a saved model |
| Updates start after | 1,000 steps |
| Update schedule | Every 50 steps, 50 gradient updates (1 update per environment step on average) |
| Max episode length | 1,000 steps (baseline runs); 4,000 steps (revised run) |
| Seed | 0 (the default; one seed so far) |
| `max_speed` / steering range | 180 km/h / ±270° |

**"Splits" in this setting.** In reinforcement learning, the training data is generated by the agent itself, so there is no train/validation/test split of a dataset. We separate data by *how* it is collected:

| Role | What it is | Status |
| --- | --- | --- |
| Training | Episodes driven with the stochastic policy while learning, stored in the replay buffer | Done (Section 3.1) |
| Validation | Separate episodes with the deterministic policy and learning turned off, used to choose checkpoints and settings | **(To be added for CI2: run and report)** |
| Test | Held-out evaluation episodes of the final selected policy, never used for tuning and never added to the replay buffer | **(To be added for CI2: run and report)** |

### 2.4 Changes from Upstream ACRL

We based our code on upstream commit `ff1b5dd` and fixed several problems before training. All runs and conditions use the fixed code.

| Problem in the original | Our fix | Why it matters |
| --- | --- | --- |
| Heading error and distance from the reference path used the car's world x and **y**. In Assetto Corsa, y is the vertical axis; the track lies in the x–z plane (the public position sample (-131.0, -0.4, -822.3) shows y near zero). | Use x and z. | Both values are observations and reward terms. With y, the agent was told how far it was from the line using its height. |
| The velocity for heading error used `velocity[0]` and `velocity[1]` (horizontal x and vertical y). | Use `velocity[0]` and `velocity[2]`. | Heading error now compares the car's direction of travel on the track with the reference path. |
| Heading error divided by the velocity's length without checking for zero, giving NaN for a stopped car. | Return heading error 0 when the car is nearly stopped. | Every episode starts from a standstill, so NaN could reach the replay buffer. **This fix contributed to the main failure in [4.1](#41-the-policy-learned-to-stand-still).** |
| The in-game app added its library paths after the imports that need them. | Add paths first; write import errors to `acrl_error.txt`. | The app loads reliably, and failures are easy to find. |
| Loading a saved model crashed (see [4.2](#42-saved-models-could-not-be-loaded)). | In `sac.py`: load `<run folder>/model.pt` with `weights_only=False` on the CPU, set `start_steps` to 0 when resuming, and always save `config.json`. | Training can continue from a saved model, and held-out evaluation of a saved model becomes possible. |

Code version: (To be added for CI2: commit or release of the code used for the overnight run)

## 3. Results and Evaluation

### 3.1 Training Runs

| Run | Episodes | Environment steps | Wall-clock time | Purpose |
| --- | --- | --- | --- | --- |
| Pilot | 5 | 5,000 | ~2.7 min | Check that the training loop runs end to end |
| Overnight | 570 | 570,000 | ~6.9 h | First real training run (baseline reward) |
| Revised reward | 189 | 119,138 | ~1.8 h | Test the revised reward, stuck termination, and 4,000-step limit |
| Revised reward, resumed | 61 | 29,900 | ~0.5 h | Continue the revised-reward run from its saved model (first test of the loading fix) |

All four logs are clean at the episode-summary level: every value is finite, and `TotalSteps` always equals the sum of episode lengths. In the first two runs, every episode is exactly 1,000 steps. This says nothing about individual telemetry samples. (To be added for CI2: repository links to all four `progress.txt` files)

**Pilot (first five episodes).** With `start_steps=10000`, every action in these 5,000 steps was random, so they serve as a preliminary random-action baseline. Net progress was 0.08–0.28% of a lap per episode, and every recorded step reward was negative.

| Episode | Episode reward | Final progress | Net progress | Avg. speed (km/h) |
| --- | --- | --- | --- | --- |
| 1 | -4,130 | 0.23% | 0.08% | 1.11 |
| 2 | -3,482 | 0.43% | 0.28% | 1.77 |
| 3 | -2,030 | 0.35% | 0.18% | 1.38 |
| 4 | -2,617 | 0.40% | 0.24% | 1.77 |
| 5 | -4,304 | 0.35% | 0.20% | 1.52 |

**Overnight run (570 episodes).** Medians by block of episodes:

| Episodes | Actions | Episode reward | Avg. speed (km/h) | Net progress (% of lap) | Episodes with any positive step reward |
| --- | --- | --- | --- | --- | --- |
| 1–10 | Random | -5,248 | 3.44 | 0.55 | 0 / 10 |
| 11–50 | Policy | -5,365 | 3.16 | 0.45 | 5 / 40 |
| 51–100 | Policy | -5,029 | 1.55 | 0.19 | 0 / 50 |
| 101–200 | Policy | -4,768 | 1.07 | 0.14 | 6 / 100 |
| 201–300 | Policy | -4,745 | 0.80 | 0.08 | 2 / 100 |
| 301–400 | Policy | -4,858 | 0.85 | 0.09 | 0 / 100 |
| 401–500 | Policy | -4,717 | 0.82 | 0.09 | 1 / 100 |
| 501–570 | Policy | -4,932 | 0.81 | 0.10 | 2 / 70 |

"Net progress" is the average per-step change in lap progress times the number of steps; forward and backward movement cancel out. "Final progress" (ACRL's `EpDist`) is the lap progress when the episode ended, which includes the ~0.15% the car starts into the lap. Medians are used because episode rewards vary widely.

**What the training results show**

- **The policy learned to move less.** Median speed fell from 3.4 km/h in the random phase to about 0.8 km/h by episode 200 and stayed there. Net progress fell from 0.55% to about 0.09% of a lap. The farthest any episode got was 2.5% of a lap, in episode 9, while actions were still random.
- **Reward improved while driving got worse.** Median episode reward rose by about 10% over the same period that progress fell about sixfold. Across all 570 episodes, reward was slightly negatively correlated with net progress (r ≈ −0.28), while speed and net progress moved together (r ≈ 0.96).
- **No episode completed a lap.** The completion rate for our main goal is 0%, and lap time is N/A. The farthest the car got was 2.5% of a lap.

![Learning curves of the overnight run](figures/overnight_learning_curves.png)

*Episode reward, average speed, and net progress for each of the 570 episodes, with a 20-episode rolling median. The shaded band marks the random-action phase (episodes 1–10). Speed and progress drop once the policy starts choosing actions, while reward stays flat or improves slightly.*

**Revised-reward run (189 episodes).** This run used our rewritten progress + speed reward, ended episodes early when the car stayed nearly stopped for 100 steps, and raised the limit to 4,000 steps. Episodes 1–23 (the first 10,000 steps) used random actions. The run's `config.json` confirms the settings: `reward_id=4`, a −40 stuck penalty, and termination after 100 consecutive steps below 2 km/h. This matches the log, where every episode's lowest step reward is about −40.4 (the stuck penalty plus the −0.4 low-speed penalty).

| Episodes | Actions | Median length (steps) | Stuck at 100 steps | Median speed (km/h) | Median net progress (% of lap) | Farthest (% of lap) | Episodes with any positive step reward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1–23 | Random | 346 | 3 | 2.1 | 0.15 | 2.2 | 14 / 23 |
| 24–50 | Policy | 453 | 1 | 2.9 | 0.28 | 1.3 | 21 / 27 |
| 51–100 | Policy | 593 | 7 | 2.1 | 0.24 | 1.5 | 34 / 50 |
| 101–150 | Policy | 537 | 8 | 2.1 | 0.23 | 2.4 | 36 / 50 |
| 151–189 | Policy | 577 | 3 | 3.2 | 0.48 | 2.4 | 34 / 39 |

Compared with the overnight baseline at the same budget (its first ~119,000 steps):

| | Overnight baseline | Revised reward |
| --- | --- | --- |
| Median speed during the policy phase | 1.9 km/h | 2.3 km/h |
| Median net progress | 0.27% | 0.26% |
| Farthest progress | 2.5% | 2.4% |
| Episodes with any positive step reward | 5 / 119 | 139 / 189 |
| End of this budget | Speed had fallen to 1.0 km/h | Speed rose to 3.2 km/h |

**What the revised run shows**

- **The collapse into standing still is gone.** Where the baseline's speed had already halved at this budget, the revised run held steady and rose in the last block (3.2 km/h, 0.48% of a lap). The upward trend after the random phase is weak, though, and not statistically reliable (Spearman ρ ≈ 0.14, p ≈ 0.08).
- **The reward now pays for moving.** 74% of episodes contain at least one positive step reward, compared with under 3% in the baseline.
- **Stuck episodes stop early.** 22 episodes ended after exactly 100 steps because the car never got moving, so far less training time went to dead episodes.
- **Every episode still ended with the car stuck.** None completed a lap or reached the 4,000-step limit. The farthest progress, 2.4% of a lap, is no better than the baseline. The completion rate is still 0%.

![Learning curves of the revised-reward run](figures/revised_reward_learning_curves.png)

*Episode length, average speed, and net progress for each of the 189 episodes, with a 20-episode rolling median. The shaded band marks the random-action phase (episodes 1–23). Unlike the baseline, speed and progress do not fall once the policy takes over, but they do not climb much either.*

**Revised reward, resumed (61 episodes).** We continued the revised-reward run from its saved model (`experiments/reward_check_2` → `experiments/reward_check_2_resume`) with the same reward and settings. Its `config.json` records `load_path=experiments/reward_check_2` and `start_steps=0`, so the loaded policy drove from the first step, and the original run's files were left untouched. As expected, the replay buffer, optimizers, and step counter started over.

| Episodes | Median length (steps) | Stuck at 100 steps | Median speed (km/h) | Median net progress (% of lap) | Farthest (% of lap) |
| --- | --- | --- | --- | --- | --- |
| 1–20 | 275 | 3 | 2.3 | 0.13 | 1.3 |
| 21–40 | 352 | 2 | 3.2 | 0.33 | 1.4 |
| 41–61 | 623 | 1 | 2.9 | 0.44 | 1.7 |
| Original run, last 39 episodes (for reference) | 577 | 3 | 3.2 | 0.48 | 2.4 |

Episode 39 is left out of these numbers because it was a false lap completion (see [4.6](#46-reversing-over-the-start-line-counts-as-a-finished-lap)).

**What the resumed run shows**

- **Loading works.** The run started from the saved weights instead of from scratch and without 10,000 random steps.
- **It dipped at first, then recovered to where the original run ended.** In the first 20 episodes, speed and progress were lower than at the end of the original run. That fits the fresh replay buffer and optimizers: the policy is retrained on a small, new buffer. By episodes 41–61 it was back to about 3 km/h and 0.4% of a lap per episode. Over the whole resumed run, speed and progress are not significantly different from the original run's policy phase (Mann–Whitney p ≈ 0.23 and 0.70).
- **The underlying problem is unchanged.** Apart from the false lap completion, every episode ended with the car stuck, the average reward per step was negative in every episode (best −0.28), and episode reward still tracked episode length almost exactly (r = −0.99). The farthest progress was 1.7% of a lap.

![Learning curves of the revised-reward run and its resumed continuation](figures/revised_reward_resumed_learning_curves.png)

*The revised-reward run (episodes 1–189) followed by the resumed run, with a 20-episode rolling median for each segment. The dashed line marks the resume. After a short dip, the resumed run returns to the level where the original run ended.*

### 3.2 Held-Out Evaluation

The numbers above come from training episodes, where the policy is stochastic and still learning. Following our evaluation plan, the baseline still needs a held-out evaluation.

| Metric | Random-action baseline | Trained SAC baseline |
| --- | --- | --- |
| Valid-lap completion rate | (To be added for CI2) | (To be added for CI2) |
| Median valid lap time | (To be added for CI2; expected N/A) | (To be added for CI2; expected N/A) |
| Median net progress per episode (% of lap) | (To be added for CI2) | (To be added for CI2) |
| Median average speed (km/h) | (To be added for CI2) | (To be added for CI2) |
| Share of episodes ending with the car stationary | (To be added for CI2) | (To be added for CI2) |

Protocol: **(To be added for CI2: number of evaluation episodes, deterministic policy with learning off, the checkpoint evaluated, and initial conditions)**

Loading saved models now works ([4.2](#42-saved-models-could-not-be-loaded)), so this evaluation can load a saved model and run it with learning turned off. **(To be added for CI2: an evaluation mode that runs the deterministic policy without updates and without adding to the replay buffer)**

### 3.3 Log Artifacts

- **`EpDist` is not distance driven.** It is the lap progress at the end of the episode, and each episode starts about 0.15–0.16% into the lap (derived from the log). We use net progress instead.
- **Step time is not fixed.** Wall-clock time per logged step was 21–92 ms in the first two runs (median ~42 ms in the overnight run) and rose to a median of 56 ms in the revised run and 61 ms in the resumed run, so the same number of steps now covers less driving per training hour. It also includes gradient updates and saving, so it is not the control interval. Measuring the control interval would need per-step timestamps.
- **One episode may not have reset properly.** Working backward from the log, episode 141 of the revised run started about 2.2% into the lap instead of the usual ~0.1%, so its 2.4% final progress is not a real gain. We need to check the reset.
- **One episode ended with a false lap completion.** Episode 39 of the resumed run shows progress jumping from about 0.2% to 99% in a single step, and the episode then ended as a completed lap. See [4.6](#46-reversing-over-the-start-line-counts-as-a-finished-lap).
- **End reasons and reward components are not logged.** The environment now returns an end reason (`lap_complete` or `stuck`) with each step, but `sac.py` does not write it to the log yet. We can't tell from the log why an episode ended or which reward term dominated. We are adding both.

### 3.4 Qualitative Examples

**(To be added for CI2: two or three concrete examples, such as a clip or screenshots of the trained car launching and braking to a stop, an x–z trajectory plot of a random-phase episode next to a late-training episode, and the per-step reward of one late episode)**

## 4. Failure Analysis

### 4.1 The Policy Learned to Stand Still

**What happened.** After the overnight run, the car mostly accelerated briefly and then braked, staying near its starting position (Section 3.1).

**Why.** Under the baseline reward, standing still is a safe local optimum:

| Reward term | Stopped car | Moving car |
| --- | --- | --- |
| Heading `cos θ − \|sin θ\|` | **Always +1**, because our zero-speed fix returns heading error 0 | +1 down to −1.41 |
| Distance from the reference path | Constant | Grows if the car drifts |
| Speed `speed / 180` | 0 | Small at low speed (0.06 at 10 km/h) |

Driving risks large penalties from bad headings and drifting (the worst step in the run was −22), while the upside at low speed is tiny. Nothing in the reward pays for forward progress, and the episode never ends early for being stuck. The agent optimized the reward correctly; the optimum was to stay still. We had flagged this side effect when making the zero-speed fix.

**Confirmation test:** (To be added for CI2: average per-step reward with no input vs. constant light throttle for ~100 steps; if standing still scores higher, the cause is confirmed)

**Fix.** Rewrite the reward so standing still is never the best option: give heading credit only while moving, add a scaled progress term, add a graded low-speed penalty, cap the distance penalty, and end episodes where the car stays stuck (with a penalty large enough that getting stuck on purpose doesn't pay). We checked the revised reward on hand-built situations: it ranks stopped < creeping < 30 km/h < 150 km/h, and reversing scores no better than stopping. In the revised run (Section 3.1) this stopped the collapse into standing still, but it did not make the car drive further; see [4.5](#45-the-revised-reward-still-makes-getting-stuck-pay).

### 4.2 Saved Models Could Not Be Loaded

**Status: fixed and verified.** The resumed run in Section 3.1 loaded `experiments/reward_check_2/model.pt`, drove with the loaded policy from its first step, and wrote to a new folder.

**What happened.** We could not resume training from the overnight run's model.

**Why.** Upstream ACRL never finished its load path:

1. It calls `logger.restore_state(load_path)`, which is not implemented in `EpisodeLogger`. Our current code adds a `restore_state` method, which stops this crash, but the method does nothing useful: it receives a folder path rather than saved state, and the logger never saves episode statistics, so there is nothing to restore.
2. `main.py` requires a folder, but `sac.py` passes that folder straight to `torch.load`, which needs the `model.pt` file inside it.
3. PyTorch 2.6+ needs `weights_only=False` because ACRL saves the whole model object.

**Fix** (all in `standalone/sac/sac.py`). Stop relying on `restore_state` and always save the new run's `config.json`; load `<folder>/model.pt` with `weights_only=False` on the CPU; and set `start_steps` to 0 when loading, so the loaded policy acts from the first step instead of taking 10,000 random actions. A resumed run must be given a new experiment name: reusing the original name overwrites its `progress.txt` and `model.pt`. An empty `experiments/Checkin_reload` folder (only an empty `progress.txt`) is consistent with an earlier resume attempt that stopped right after the logger created its folder. We had written these fixes earlier, but they were not in the copy of the code that ran overnight. Even with the fix, only the networks are saved; the replay buffer, optimizers, and step counter start over, so a resumed run counts as a separate run segment.

### 4.3 Episodes Are Too Short to Finish a Lap

**What happened.** Episodes are capped at 1,000 steps, about 42 seconds at the observed step time.

**Why it matters.** Assetto Corsa's built-in AI laps Silverstone 1967 in this car, with the same set of assists but faster tires, in about 85 seconds (1:25.454), which is roughly 2,000 steps, and the 1:25.454 was set after the AI had been given a warm up/out lap, so the step count for that is actually 4000. Even a perfect policy could not complete a lap within 1,000 steps, so our main goal was impossible by construction.

**Fix.** Raise the limit to about 4,000 steps, or base it on wall-clock time since step time varies.

### 4.4 The Other ACRL Reward Functions Have the Same Problem

Before switching rewards, we reviewed ACRL's four unused reward functions. None works as is:

| Function | Rewards | Main problems |
| --- | --- | --- |
| `_get_reward_1` | Speed | Crashes at 10% of a lap (empty `progress_weights` list); a flat −0.8 below ~40 km/h treats 0 and 30 km/h the same; after the first off-track moment, every remaining step is −1 |
| `_get_reward_2` | Absolute lap position + speed | Same problems, and it pays for *where* the car is rather than how far it moved |
| `_get_reward_3` | Change in progress | Same problems, and the progress reward is ~2,000× smaller than the penalties |
| `_get_reward_4` | Change in progress + speed | Same shared problems; the progress term is negligible without scaling |

The flat penalty below ~40 km/h is especially relevant: like the baseline reward, it gives the agent no reason to speed up from a standstill. We rewrote all five reward functions to fix these issues and will use the revised progress + speed reward (based on `_get_reward_4`) as the starting point.

### 4.5 The Revised Reward Still Makes Getting Stuck Pay

**What happened.** In the revised run, every one of the 189 episodes ended because the car got stuck (Section 3.1).

**Why.** Leaving out the final penalty step, the average reward per step was negative in every episode: the median was −0.36, and even the best episode averaged −0.24. So episode reward depends almost entirely on episode length (r = −0.99):

| Episode type | Median episode reward |
| --- | --- |
| Stuck right away (100 steps) | −79 |
| Drove for 1,000+ steps | −528 |

By total reward, getting stuck immediately is the best outcome. Discounting turns this into a near-tie rather than a clear win for driving: with γ = 0.99, getting stuck now is worth about −40, and driving on at −0.36 per step is worth about −36. The agent has almost no incentive to keep driving.

Exploration makes this worse. Throttle and brake share one action axis, so uniformly random actions brake half the time. In the random phase the car barely moved (median 2.1 km/h), so the replay buffer holds very few examples of driving at real speed.

**Fix.**

1. Make good driving earn a clearly positive reward per step (for example at 30–60 km/h on the right heading), by scaling up the progress term or shrinking the constant per-step penalties.
2. Make getting stuck clearly worse than driving, for example a stuck penalty of −100 instead of −40, and recheck the discounted comparison above.
3. Bias the random phase toward the throttle, for example sampling the throttle/brake axis mostly from [−0.2, 1] instead of [−1, 1].

### 4.6 Reversing Over the Start Line Counts as a Finished Lap

**What happened.** Episode 39 of the resumed run ended as a completed lap after 463 steps, although the car had covered only a tiny part of the track. In one step, lap progress jumped from about 0.002 to 0.99.

**Why.** The car starts just past the start/finish line. If it rolls backward over the line, lap progress wraps from about 0 to about 0.99. The environment's lap check is `lap_done = lap_count > 1.0 or track_progress >= progress_goal`, with `progress_goal = 0.99`, so this counts as finishing the lap. The episode then ends without the −40 stuck penalty: the worst step in that episode was −5.4 (the clipped backward-progress penalty plus the low-speed penalty), compared with about −40.4 in every other episode. The agent did not exploit this yet, but ending an episode by reversing over the line is much cheaper than getting stuck, so it could learn to.

This also corrupts the logs. `EpDist` and `DistHigh` jump to 99%, and a "lap complete" end reason would be reported for a lap that never happened.

**Fix.** Count a lap as complete only when the car reaches the end of the lap going forward, and treat a backward crossing as a failure. In `step()` in `ac_environment.py`:

```python
prev_progress = observation[7]
crossed_backward = track_progress - prev_progress > 0.5   # wrapped from ~0 to ~1
lap_done = lap_count > 1.0 or (track_progress >= progress_goal and not crossed_backward)
wrong_way = crossed_backward
terminated = lap_done or stuck or wrong_way
# ...and add a penalty at least as large as penalty_stuck when wrong_way,
# with end_reason "wrong_way"
```

### 4.7 Next Steps Toward Check-In 3

1. ~~Put the loading fix into the code we run and confirm a resumed run acts with the loaded policy from its first step.~~ Done (Section 3.1).
2. Fix the lap-completion check so reversing over the start line ends the episode as a failure, not a finished lap (4.6).
3. Log reward components, end reasons, and per-step timestamps.
4. Run the confirmation test for 4.1.
5. Revise the reward so driving earns a positive reward and getting stuck clearly costs more (4.5), and bias early exploration toward the throttle. This changes the baseline, so it will apply to every condition (or become experiment A/C itself).
6. Check and fix the episode reset (Section 3.3) before the next long run.
7. Retrain with at least 3 seeds and run the held-out evaluation (Section 3.2), loading each saved model with learning turned off.
8. Once the car completes laps reliably (Stage 1), move on to Stage 2: compare the median valid lap time with the built-in AI's 1:25.454.

## 5. Data and Evaluation Plan (from Check-In 1)

### 5.1 Public Data

The public [track_data directory](https://github.com/Jurredr/ACRL/tree/master/track_data) (GPL-3.0, like the rest of ACRL) contains:

| File | Size and structure | Main contents |
| --- | --- | --- |
| `car.csv` | 15,681 rows, 17 columns | Speed, RPM, velocity, acceleration, position, timestamps |
| `lap.csv` | 15,681 rows, 10 columns | Lap progress, lap count, timing, sector, invalid flag |
| `tick.csv` | 15,681 rows, 3 columns | Lap location, world location, velocity |
| `spline_points.csv` | 2 × 1,000 array | Reference path coordinates (used by our environment) |

The CSVs describe the same recording of roughly one lap (about 523 seconds, sampled every 33 ms on median, 2–220 ms range) and should not be counted as independent datasets. They contain no steering or pedal inputs or rewards, so they support an audit but not behavioral cloning. The invalid flag is `False` in every row, which doesn't prove the car stayed on track. Raw `speed` (mean 32.8) needs its units confirmed.

Audit figures (trajectory, speed distribution, sampling intervals, section coverage, data-quality table): (To be added)

### 5.2 Metrics

| Role | Measure |
| --- | --- |
| Stage 1 (main) | Valid-lap completion rate in held-out evaluation; environment steps until the first valid lap |
| Stage 1 progress (before any lap is completed) | Farthest progress per episode; rates of reaching one sector, two sectors, and a full lap |
| Stage 2 (stretch) | Median valid lap time, and its gap to the built-in AI's 1:25.454 |
| Failure diagnostics | Stuck episodes, off-track events, and where failures happen |
| Cost | Gradient updates and wall-clock training time |

Lap times are always reported with completion rates. A run that never completes a lap has lap time **N/A**; a run that never reaches a target is reported as **not reached within budget**.

### 5.3 Protocol

| Item | Plan |
| --- | --- |
| Training seeds | At least 3 independent seeds per condition; (To be added: final list) |
| Evaluation interval | (To be added: in environment steps) |
| Validation / test episodes | (To be added: counts) |
| Evaluation actions | Deterministic policy, learning off |
| Checkpoint selection | (To be added: final checkpoint or validation-based rule) |

Every condition uses the same budget, update schedule, shared fixes, and evaluation setup. When reward functions differ across conditions, we compare driving outcomes rather than raw rewards.

### 5.4 Success Criteria

| Criterion | Value or rule to finalize |
| --- | --- |
| Stage 1 reached | (To be added: e.g. at least one valid lap in held-out evaluation, then a minimum completion rate) |
| Moving on to Stage 2 | (To be added: the completion rate required before we focus on lap time) |
| Stage 2 reached | Median valid lap time at or below the built-in AI's 1:25.454, with the completion rate above the Stage 1 minimum |
| Meaningful improvement between conditions | (To be added: in completion rate, steps to first lap, or lap time) |

A negative result is still worth reporting: a fair experiment with a clear account of its limits is a useful outcome.

## Milestones

| Stage | Deliverable | Deadline |
| --- | --- | --- |
| Check-In 2 | Working SAC baseline, training results, failure analysis | (To be added) |
| Revised baseline | Loading fix (done), revised reward, longer episodes (first attempt done: Section 3.1) | (To be added) |
| Stage 1 | First valid lap, then a reliable completion rate | (To be added) |
| Check-In 3 | Baseline vs. revised reward, multiple seeds, held-out evaluation | (To be added) |
| Stage 2 (stretch) | Lap time compared with the built-in AI's 1:25.454 | (To be added) |
| Final evaluation | Frozen test results, uncertainty, failure analysis, limitations | (To be added) |

## AI Disclosure

Claude (Anthropic) has been used for grammar improvement on README and code debugging ideas.
