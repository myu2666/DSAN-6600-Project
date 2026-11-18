# DSAN-6600-Project

Semester Project for DSAN 6600: Deep Learning and Neural Networks

We use [ACRL](https://github.com/Jurredr/ACRL), a Soft Actor-Critic (SAC) implementation for Assetto Corsa, to train an agent that can drive a lap of a race track on its own. If it gets there, our next target is the lap time set by Assetto Corsa's built-in AI on the same car and track.

**Status (Check-In 3):** (To be added: the extension, its effect compared with the SAC baseline with shared fixes, and where the agent now stands relative to Stage 1)

## Check-In 3 at a Glance

| Rubric item | Where to look | Status |
| --- | --- | --- |
| Advanced extension | [3. Advanced Extension](#3-advanced-extension) | (To be added) |
| Comparison with the CI2 baseline | [4. Comparison with the Baseline](#4-comparison-with-the-baseline) | (To be added) |
| Ablation / controlled comparison | [5. Ablation](#5-ablation) | (To be added) |
| Failure analysis | [6. Failure Analysis](#6-failure-analysis) | (To be added) |
| Progress video (~5 min) | [Progress Video](#progress-video) | (To be added) |
| Demo notebook | (To be added: link) | (To be added) |
| Code | [1.3 Repository and How to Run](#13-repository-and-how-to-run) | (To be added) |

**Key result:** (To be added: completion rate, net progress, and farthest progress of the baseline with shared fixes vs. the extension under the same evaluation, number of seeds, and whether the difference is larger than the variation between seeds)

The full Check-In 2 write-up (training runs, log checks, and failure analysis of the original baseline) is in [check-in-2.md](check-in-2.md).

## Progress Video

**Link:** (To be added)

| Time | Content |
| --- | --- |
| 0:00–0:30 | The problem left at Check-In 2 and our hypothesis |
| 0:30–1:30 | The extension and how it plugs into SAC |
| 1:30–3:30 | The car driving in the game, and the comparison with the baseline |
| 3:30–4:30 | Ablation result |
| 4:30–5:00 | Where the extension still fails, and limitations |

## 1. Project Overview

### 1.1 Goal

AI opponents largely decide how challenging and fun a racing simulator feels. Unlike chess or Go, racing needs continuous control of steering, throttle, and brake while reacting to how the car moves, and one small mistake can ruin a corner or invalidate a lap. We want to see how far an agent that learns only by driving can get.

| Stage | Goal | How we measure it |
| --- | --- | --- |
| **1. Complete a lap (main goal)** | From the standard start position, the agent drives forward to the finish line on its own, and the game never marks the lap invalid. | Lap completion rate in held-out evaluation; environment steps until the first completed lap |
| **2. Catch the built-in AI (stretch goal)** | Once the agent completes laps reliably, close the gap to Assetto Corsa's built-in AI in the same car on the same track. | Median time of full timed laps compared with the built-in AI's lap, re-recorded on our tires (the current reference, **1:25.454**, was set on faster tires) |

**What a completed lap means for Stage 1.** Every episode starts with the car at rest about 0.15% of a lap past the start/finish line. A Stage 1 lap is the drive from there, forward around the track, to the finish line: about 99.85% of the lap, from a standing start. It is not a full timed lap (line to line at racing speed), which only matters for Stage 2. The evaluation in [4.1](#41-evaluation-protocol) applies exactly this definition.

At Check-In 2, the SAC baseline never completed a lap: the original reward led the car toward standing still, and with a revised reward every episode still ended with the car stuck within the first 2.5% of the lap. For Check-In 3 we fixed the bugs found at Check-In 2 for every condition alike ([2.3](#23-shared-fixes-since-check-in-2)) and added one extension on top of SAC ([3](#3-advanced-extension)).

### 1.2 Scope

One car, one track, no opponents, fixed driving conditions.

| Setting | Value |
| --- | --- |
| Car and track | Ferrari 458 GT2 on Silverstone 1967 (the defaults in `main.py`) |
| Session settings | (To be added: tires, assists, weather, grip, fuel) |
| Hardware | (To be added) |
| Training budget | Check-In 2: about 720,000 environment steps over four runs. Check-In 3: (To be added) |

### 1.3 Repository and How to Run

| Folder or file | Contents |
| --- | --- |
| `README.md` | This write-up (Check-In 3) |
| `check-in-2.md` | Check-In 2 write-up, kept for reference |
| `figures/` | All plots in both write-ups |
| `logs/<run name>/` | Episode logs (`progress.txt`) and evaluation results of every run |
| (To be added) | Demo notebook with the comparison results |
| (To be added) | Code (layout below) |

```
(To be added: final code layout, including the extension's files)
ACRL.py                         # In-game app: sends telemetry from Assetto Corsa over a socket
standalone/
├── main.py                     # Entry point: settings, hyperparameters, start or resume training
├── evaluate.py                 # Held-out evaluation without learning
├── ac_controller.py            # Virtual Xbox controller (vgamepad)
├── ac_socket.py                # Socket connection to the in-game app
└── sac/
    ├── ac_environment.py       # Gymnasium environment: observations, reward, termination
    ├── sac.py                  # SAC agent, replay buffer, training loop
    ├── core.py                 # Actor and critic networks
    └── utils/                  # Reference-path geometry, logging
track_data/spline_points.csv    # Reference path (from upstream ACRL)
```

**Setup.** Install Assetto Corsa and follow the upstream [setup instructions](https://github.com/Jurredr/ACRL#readme), copy the in-game app into `assettocorsa/apps/python/ACRL`, enable it, and start a practice session with the car and track above.

**Commands** (from `standalone/`, with the game running):

| Task | Command |
| --- | --- |
| Train the SAC baseline with shared fixes | (To be added) |
| Train the extension | (To be added) |
| Train the ablation condition | (To be added) |
| Evaluate a saved model | `python evaluate.py --model experiments/<run name> --episodes 20` |
| Evaluate random actions | `python evaluate.py --random --episodes 20 --seed 0` |

## 2. Baseline

### 2.1 Conditions Compared

"The CI2 baseline" can mean two different things, so we keep them apart:

| Name in this write-up | What it is | Role |
| --- | --- | --- |
| **Original CI2 checkpoint** | The model saved at Check-In 2 (`experiments/reward_check_2_resume/model.pt`), trained with the Check-In 2 code | Reference only: shows how far we are from where we stood at Check-In 2. Its training conditions differ from the others. |
| **SAC baseline (shared fixes)** | SAC without the extension, retrained with every shared fix in [2.3](#23-shared-fixes-since-check-in-2) | **Main comparison.** Differs from the extension only by the extension. |
| **SAC + extension** | The same, with the extension added | The condition being tested |
| Random actions | Uniform random actions, no learning | Floor for every metric |

Differences between the original CI2 checkpoint and the SAC baseline (shared fixes) come from the shared fixes. Differences between the SAC baseline (shared fixes) and SAC + extension come from the extension. (To be added: whether the original CI2 checkpoint was evaluated, and the runs and seeds behind each condition)

### 2.2 Shared Setup

The task, network, and SAC settings below are the same for every trained condition unless Section 3 says otherwise. Check-In 2 has the full description ([check-in-2.md, Section 2](check-in-2.md#2-neural-baseline-sac)).

| Component | Setting |
| --- | --- |
| Observation (10 values) | Lap progress, speed, world x/y/z, lap-invalid flag, lap count, previous lap progress, heading error and distance relative to the reference path |
| Action (2 values in [−1, 1]) | Combined throttle/brake, steering |
| Reward | (To be added: the reward used by every condition; at Check-In 2, `reward_id=4`) |
| Episode end | Finish line reached, car stuck (100 steps below 2 km/h), or (To be added) step limit |
| Policy | MLP 10 → 256 → 256, squashed Gaussian |
| Critics | Two MLPs 12 → 256 → 256 → 1, Polyak-averaged targets (0.999) |
| SAC settings | Adam, learning rate 1e-3, γ = 0.99, α = 0.2 (fixed), batch 32, buffer 10⁶, (To be added: random-action steps and update schedule) |

### 2.3 Shared Fixes Since Check-In 2

These fixes change the environment or the training loop, so they apply to every trained condition. Any improvement they cause is credited to the fixes, not to the extension.

| Fix | Problem it addresses (Check-In 2) | Status |
| --- | --- | --- |
| Model loading | Saved models could not be loaded ([CI2 4.2](check-in-2.md#42-saved-models-could-not-be-loaded)) | Done at Check-In 2 |
| Lap check | Reversing over the start line counted as a finished lap ([CI2 4.6](check-in-2.md#46-reversing-over-the-start-line-counts-as-a-finished-lap)) | (To be added) |
| Update timing | The car drove blind during gradient updates ([CI2 4.7](check-in-2.md#47-the-car-drives-blind-during-gradient-updates)) | (To be added) |
| Reset | One episode started 100 m down the track ([CI2 3.3](check-in-2.md#33-log-artifacts)) | (To be added) |
| Logging | End reasons, reward components, and step timestamps were not logged | (To be added) |
| Reward / exploration | Driving never earned a positive reward on average ([CI2 4.5](check-in-2.md#45-the-revised-reward-may-still-give-too-little-incentive-to-drive)) | (To be added: changed for all conditions, or not changed) |
| (To be added) | (To be added) | (To be added) |

## 3. Advanced Extension

### 3.1 What We Added and Why

| | |
| --- | --- |
| Extension | (To be added) |
| Rubric category | (To be added: e.g. different architecture, multimodal input, agentic layer, robustness or data-centric improvement) |
| What it adds beyond Check-In 2 | (To be added: the new neural component or method, compared with the plain SAC MLP and the bug fixes in 2.3) |
| Scope | (To be added: what is in and out of this extension) |

**Motivation.** (To be added: the model behavior from Check-In 2 this targets, with a link to the relevant part of check-in-2.md)

**Hypothesis.** (To be added: the expected change in a metric from 4.1, under the same budget)

### 3.2 Design

(To be added: diagram or short description of how the extension connects to the SAC policy and critics)

| Component | SAC baseline (shared fixes) | SAC + extension |
| --- | --- | --- |
| Observation | 10 values | (To be added) |
| Policy network | MLP 10 → 256 → 256 | (To be added) |
| Critic networks | Two MLPs 12 → 256 → 256 → 1 | (To be added) |
| Other components | None | (To be added) |
| Parameters | ~208,000 | (To be added) |

### 3.3 Implementation

| Item | Details |
| --- | --- |
| Files added or changed | (To be added) |
| How to turn it on | (To be added) |
| Extra hyperparameters | (To be added) |
| Extra compute | (To be added: time per step or per update, compared with the baseline) |

### 3.4 Evidence That It Runs

(To be added: learning curve, driving trace, or log excerpt from a training run of the extension, with a link to its log)

## 4. Comparison with the Baseline

### 4.1 Evaluation Protocol

Training episodes use a stochastic policy that is still learning, so all conditions are compared on separate held-out episodes instead.

| Item | Setting |
| --- | --- |
| Tool | `standalone/evaluate.py`: loads a saved model and drives with it; no gradient updates and no replay buffer, so nothing it sees reaches training |
| Conditions | The four in [2.1](#21-conditions-compared) |
| Actions | Trained policies: deterministic mean action. Random: uniform over [−1, 1] |
| Episodes | (To be added: per condition and seed) |
| Seeds | (To be added: training seeds per condition, evaluation seed) |
| Checkpoint | (To be added: final checkpoint, or a rule fixed before looking at test results) |
| Start | Standard reset and scripted launch, as in training |
| Environment | Shared setup in [2.2](#22-shared-setup) and fixes in [2.3](#23-shared-fixes-since-check-in-2). The script also applies the stuck rule after 99% progress, where the environment would otherwise report the lap as done. |

**Completed lap.** A Stage 1 lap as defined in [1.1](#11-goal): the game's lap count goes up because the car crossed the finish line going forward, the game never marked the lap invalid, and lap progress never jumped by more than half a lap in one step (a backward crossing of the start line ends the episode as a failure). Episodes where the environment reports 99% progress but the car never crosses the line do not count.

**Metrics.**

| Metric | Use |
| --- | --- |
| Lap completion rate | Main metric (Stage 1) |
| Net progress per episode, farthest progress | Progress before any lap is completed |
| Share of episodes ending stuck, and where they end | Failure analysis |
| Average speed | Secondary |
| Approximate lap time (completed laps only) | Reported, not compared: it is the game's timer at the last sample before the line, from a standing start |
| Training wall-clock time | Cost |

### 4.2 What Was Held Fixed

| Item | SAC baseline (shared fixes) | SAC + extension | Same? |
| --- | --- | --- | --- |
| Code version and shared fixes | (To be added) | (To be added) | (To be added) |
| Reward and termination | (To be added) | (To be added) | (To be added) |
| Training budget (environment steps) | (To be added) | (To be added) | (To be added) |
| SAC hyperparameters and update schedule | (To be added) | (To be added) | (To be added) |
| Training seeds | (To be added) | (To be added) | (To be added) |
| Evaluation | Section 4.1 | Section 4.1 | Yes |
| Hardware | (To be added) | (To be added) | (To be added) |

(To be added: anything that could not be matched, and its likely effect)

### 4.3 Results

| Metric | Random actions | Original CI2 checkpoint | SAC baseline (shared fixes) | SAC + extension |
| --- | --- | --- | --- | --- |
| Lap completion rate | (To be added) | (To be added) | (To be added) | (To be added) |
| Median net progress per episode (% of lap) | (To be added) | (To be added) | (To be added) | (To be added) |
| Farthest progress (% of lap) | (To be added) | (To be added) | (To be added) | (To be added) |
| Share of episodes ending stuck | (To be added) | (To be added) | (To be added) | (To be added) |
| Median average speed (km/h) | (To be added) | (To be added) | (To be added) | (To be added) |
| Training wall-clock time | – | – | (To be added) | (To be added) |

| Seed | SAC baseline (shared fixes): completion rate / median net progress | SAC + extension: completion rate / median net progress |
| --- | --- | --- |
| (To be added) | (To be added) | (To be added) |

(To be added: learning curves of both trained conditions, e.g. `figures/ci3_learning_curves.png`)

### 4.4 What Improved, What Got Worse, and Why

| | Original CI2 checkpoint → SAC baseline (shared fixes) | SAC baseline (shared fixes) → SAC + extension |
| --- | --- | --- |
| Credited to | Shared fixes | Extension |
| Improved | (To be added) | (To be added) |
| Worse or unchanged | (To be added) | (To be added) |

**Why.** (To be added: explanation, marked as a hypothesis where untested)

**How sure we are.** (To be added: number of seeds, and the difference compared with the spread between seeds)

## 5. Ablation

One controlled change, with everything else held at the Section 4.2 settings.

| | |
| --- | --- |
| Question | (To be added) |
| Change | (To be added: the one component removed, replaced, or varied) |
| Held fixed | Budget, seeds, shared fixes, evaluation (Section 4.1) |

| Metric | (To be added: condition A) | (To be added: condition B) |
| --- | --- | --- |
| Lap completion rate | (To be added) | (To be added) |
| Median net progress per episode (% of lap) | (To be added) | (To be added) |
| Farthest progress (% of lap) | (To be added) | (To be added) |
| Share of episodes ending stuck | (To be added) | (To be added) |

**Interpretation.** (To be added)

## 6. Failure Analysis

### 6.1 Failure Patterns of the Extension

| Pattern | Episodes (share) | Where on the track | Example (run, seed, episode) | Likely cause |
| --- | --- | --- | --- | --- |
| (To be added) | (To be added) | (To be added) | (To be added) | (To be added) |

(To be added: how end reasons and end positions were counted)

### 6.2 Concrete Examples

(To be added: trajectory and speed plots of two or three episodes, e.g. `figures/ci3_failure_examples.png`, with one note per episode)

### 6.3 Check-In 2 Failures Now

| CI2 failure | Addressed by | Evidence | Remaining issue |
| --- | --- | --- | --- |
| Drift toward standing still ([CI2 4.1](check-in-2.md#41-the-policy-learned-to-stand-still)) | (To be added: shared fix / extension / neither) | (To be added) | (To be added) |
| Every episode ends stuck ([CI2 4.5](check-in-2.md#45-the-revised-reward-may-still-give-too-little-incentive-to-drive)) | (To be added) | (To be added) | (To be added) |
| False lap from reversing over the start line ([CI2 4.6](check-in-2.md#46-reversing-over-the-start-line-counts-as-a-finished-lap)) | Shared fix (lap check) | (To be added) | (To be added) |
| Driving blind during gradient updates ([CI2 4.7](check-in-2.md#47-the-car-drives-blind-during-gradient-updates)) | Shared fix (update timing) | (To be added) | (To be added) |
| Faulty reset ([CI2 3.3](check-in-2.md#33-log-artifacts)) | Shared fix (reset) | (To be added) | (To be added) |

### 6.4 Implications

(To be added)

### 6.5 Remaining Limitations

- One car, one track, and fixed conditions, so the results say nothing about other cars, tracks, or conditions.
- Training runs in real time on one laptop, which limits the budget and the number of seeds.
- (To be added: number of seeds and its effect on the conclusions)
- (To be added: limitations of the extension)

## 7. Next Steps

(To be added)

## Milestones

| Stage | Deliverable | Deadline |
| --- | --- | --- |
| Check-In 2 | Working SAC baseline, training results, failure analysis ([check-in-2.md](check-in-2.md)) | 10/28/2026 |
| Check-In 3 | Advanced extension, comparison with the baseline, ablation, failure analysis, progress video | (To be added) |
| Stage 1 | First completed lap, then a reliable completion rate | TBD |
| Stage 2 (stretch) | Full timed laps compared with the built-in AI's lap re-recorded on our tires | TBD |
| Final evaluation | Frozen test results, uncertainty, failure analysis, limitations | TBD |

## AI Disclosure

Claude (Anthropic) was used to draft and edit the README and its structure, write analysis and plotting scripts and `evaluate.py`, and suggest debugging ideas. (To be added: confirm or adjust this to match how the team used AI tools)
