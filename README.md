# DSAN-6600-Project

Semester Project for DSAN 6600: Deep Learning and Neural Networks

We use [ACRL](https://github.com/Jurredr/ACRL), a Soft Actor-Critic (SAC) implementation for Assetto Corsa, to test whether one small design change helps a racing agent learn faster or set faster valid laps.

**Status:** This is a project proposal with a first look at the public data. We have not run our own training or evaluation yet. Items in parentheses mark decisions, evidence, or results we still need to fill in.

## 1. Research Goals and Scope

### Motivation

AI opponents are a big part of many video games, and in racing simulators they largely decide how challenging and fun a race feels. We want to see whether an agent that learns by driving can reach competitive lap times.

AI has already beaten top human players in chess and Go. Those games have huge search spaces, but the moves are discrete and the rules are fixed. Racing is a different kind of problem. The agent has to control steering, throttle, and brake continuously while reacting to how the car moves, and one small mistake can ruin a corner or invalidate a lap.

Game opponents do not necessarily learn from experience. Our project focuses on an agent that improves through interaction with the simulator. Since our time and compute are limited, we will test one change carefully instead of modifying several parts of the system at once.

Our long-term interest is an agent that can set competitive lap times. In this project, we will measure progress toward that goal against a reproducible ACRL baseline under matched conditions. Human or built-in AI lap times may provide additional context when comparable results are available. Beating our own baseline alone would not mean the agent drives at a human level.

### Main Quest: Learn Faster or Drive Faster

| Objective | Question | Main measure |
| --- | --- | --- |
| **Efficiency** | Can the agent reach a target lap time with fewer interactions? | Environment steps needed to hit both a lap-time target and a minimum completion rate |
| **Efficacy** | Can the agent drive faster with the same training budget? | Median valid lap time, as long as the completion rate stays above the minimum |

We will pick one of these as the primary objective and report the other as a secondary result. Unless stated otherwise, "iterations" means environment steps. We will also log gradient updates and wall-clock time.

| Decision | Current plan |
| --- | --- |
| Primary objective | (To be added: primary objective: efficiency or efficacy) |
| Initial intervention | (To be added: first experiment from A, B, or C and the reason for choosing it) |
| Initial hypothesis | (To be added: expected change and its rationale) |
| Decision deadline | (To be added: dates for completing the pilot and finalizing the experiment settings) |

### Sub-Quests: Candidate Interventions

| Experiment | Hypothesis | Controlled comparison |
| --- | --- | --- |
| **A. Adjust the reward** | A strong centerline penalty may make driving more stable but push the agent away from faster racing lines. | Compare two penalty weights, keeping all other reward terms and settings the same. |
| **B. Test the observations** | Absolute coordinates may help on a single track, but we don't know how much they actually contribute. | Mask the absolute coordinates while keeping the network architecture and other inputs unchanged. |
| **C. Diagnose costly failures** | Repeated failures may waste training steps or keep lap times from improving. | Find one failure pattern in validation runs and test one targeted fix. |

All three experiments serve the main objective; we are not running three separate projects. The observation ablation alone does not establish how well the agent would generalize to another track.

### Success Criteria

The baseline will include the shared implementation fixes needed for valid experiments, and every condition will get the same fixes before we compare them.

For efficiency, we will set the lap-time target after the baseline pilot and lock it in before testing any intervention. For efficacy, we will compare lap times after equal training budgets. A faster lap time doesn't count as an improvement if the completion rate drops below the limit we set in advance.

| Criterion | Value or rule to finalize |
| --- | --- |
| Target lap time for efficiency | (To be added: a target time or a rule for deriving it from pilot results) |
| Minimum valid-lap completion rate | (To be added: minimum completion rate and how it will be calculated) |
| Meaningful efficiency improvement | (To be added: target reduction in environment steps) |
| Meaningful efficacy improvement | (To be added: target reduction in lap time) |
| Acceptable completion-rate decrease | (To be added: allowable decrease in percentage points) |

Our goal is better performance, but a negative result is still worth reporting. Even if we miss the target, a fair experiment with a clear account of its limits is a useful outcome.

### Core Scope and Feasibility

We will use one car, one track, no opponents, and fixed driving conditions.

| Setting | Planned configuration |
| --- | --- |
| Car and track | (To be added: car and track names) |
| Session settings | (To be added: fixed tires, assists, weather, grip, fuel, and other session conditions) |
| Hardware and runtime | (To be added: Windows environment, CPU/GPU/RAM, Python version, and key package versions) |
| Available compute time | (To be added: available runtime per week and total compute budget) |
| Pilot budget | (To be added: pilot step budget or time limit) |
| Main experiment budget | (To be added: training steps per condition and seed, and total number of runs) |

If the baseline can't finish a lap reliably, our first milestone becomes getting a working baseline. Measures like how often the agent reaches each section, or how far it gets before failing, will help us diagnose problems, but lap time stays the main evidence.

If we still can't get reliable laps by (To be added: date for reviewing project scope), we will write up the limitation and agree on a narrower goal instead of claiming a lap-time improvement we couldn't measure.

### Stretch Goal: Transfer

If we finish the main comparison, we would like to try a second track with the same car. We expect this may be easier than switching cars because the car's underlying characteristics remain more consistent. A different layout, surface, elevation profile, and braking points could still make transfer difficult.

- **Zero-shot:** run the trained policy on the new track with no further training.
- **Fine-tuning:** compare continuing training from the existing policy with training from scratch, using the same target and budget.

We will document any new reference path or track preparation. Here, "zero-shot" means no extra training; the agent may still receive basic information about the new track. Transferring to a different car is out of scope unless time allows.

## 2. Data Access and Documentation

### Available Data

The public [track_data directory](https://github.com/Jurredr/ACRL/tree/master/track_data) contains the files below. Row counts exclude headers.

| File | Size and structure | Main contents | Intended use |
| --- | --- | --- | --- |
| `car.csv` | 15,681 rows, 17 columns | Speed, RPM, velocity, acceleration, position, timestamps, and other vehicle fields | Vehicle-state audit |
| `lap.csv` | 15,681 rows, 10 columns | Lap progress, lap count, timing, sector, invalid flag, timestamp | Progress and timing audit |
| `tick.csv` | 15,681 rows, 3 columns | Lap location, world location, velocity | Trajectory inspection |
| `spline_points.csv` | 2 × 1,000 numerical array, no header | Two coordinate sequences | Reference-path inspection |

The CSV files contain related records and should not be counted as independent datasets. Equal row counts don't guarantee that the rows line up. Before combining them, we will check timestamp alignment between `car.csv` and `lap.csv` and verify how the derived rows in `tick.csv` correspond to those records.

The files don't include steering or pedal inputs, rewards, or full state–action–next-state transitions. They are useful for an initial audit but can't be used directly for behavioral cloning.

### Access Instructions and Provenance

1. Open the [ACRL repository](https://github.com/Jurredr/ACRL).
2. Select **Code → Download ZIP**, or clone the repository.
3. Look inside `track_data/`.
4. To collect new data from the simulator, follow the upstream [setup instructions](https://github.com/Jurredr/ACRL#readme).

- Upstream commit used for experiments: (To be added: pinned commit SHA)
- Local code version: (To be added: project commit or release)
- Data snapshot and audit script/notebook: (To be added: file paths or links)

### Simulator Access Status

Downloading the public files and getting the simulator working are separate steps. We won't call the data collection pipeline ready until the checks below pass.

| Check | Status and evidence |
| --- | --- |
| Assetto Corsa runs with the selected car and track | (To be added: verification status and runtime environment) |
| ACRL receives live telemetry | (To be added: sample log or verification result) |
| Actions reach the game | (To be added: control-test results) |
| Automatic reset works consistently | (To be added: repeated reset-test results) |
| Complete transitions can be saved and loaded | (To be added: an actual sample file and confirmation that it can be read) |

### New Collection Schema

| Field | Meaning |
| --- | --- |
| `observation`, `next_observation` | State vectors before and after the action |
| `action` | Throttle/brake and steering commands actually sent, each in [-1, 1] |
| `reward` | Reward under the recorded experiment configuration |
| `terminated`, `truncated` | Episode-ending and time-limit flags, stored separately |
| `end_reason` | Completion, driving failure, timeout, or infrastructure error |
| `timestamp`, `control_interval` | Timing information with documented units |
| `session_id`, `episode_id`, `seed` | Identifiers for grouping and reproducibility |
| `config_id` | Link to the car, track, policy, reward, and runtime settings |

Final field definitions, units, storage format, and exclusion rules: (To be added: link to the data dictionary and preprocessing rules)

Actual collected volume: (To be added: numbers of sessions, episodes, and transitions, and recording duration)

### Licensing and Checkpoints

The upstream repository is released under the [GPL-3.0 license](https://github.com/Jurredr/ACRL/blob/master/LICENSE). We will credit the original authors and note separately which terms apply to the code and which apply to the data.

Data usage/sharing notes: (To be added: findings on any separate public-data usage terms and the sharing scope for newly collected data)

We did not find a ready-to-use trained ACRL checkpoint, so we plan to train from scratch. If we later bring in an outside checkpoint, we will check where it came from, how it was trained, and whether it is compatible, and record it as a change to the plan.

## 3. Data Audit and EDA

### Preliminary Measurements

These numbers describe the public files, not an agent we trained.

| Item | Observed value | Implication |
| --- | --- | --- |
| Recorded time points | 15,681 | Small dataset; consecutive rows are highly correlated |
| Duration | About 523.3 seconds | Roughly 8.7 minutes of driving |
| Recording interval | Median 33 ms; range 2–220 ms | Sampling is irregular |
| Lap count | 1 throughout the record | The record appears to cover one lap, rather than multiple independent attempts |
| Lap progress | About 0.000108–0.999985 | Covers almost the full lap |
| Raw `speed` column | Mean 32.79; range 17.25–65.76 | Units need to be confirmed before reading this as physical speed |
| Invalid-lap flag | `False` in all 15,681 rows | No invalid-lap examples |
| Steering/throttle/brake labels | Absent | We need to collect new data to study actions |

An invalid flag of `False` everywhere doesn't prove the car stayed on track the whole time. We will track off-track events separately in our own evaluations.

Sources: [car.csv](https://github.com/Jurredr/ACRL/blob/master/track_data/car.csv), [lap.csv](https://github.com/Jurredr/ACRL/blob/master/track_data/lap.csv), and [tick.csv](https://github.com/Jurredr/ACRL/blob/master/track_data/tick.csv).

### Representative Sample

The first row of the public data (rounded):

| Field | Value |
| --- | --- |
| Lap progress | 0.000108 |
| World position | (-131.023, -0.411, -822.346) |
| Raw speed value | 27.807 |
| Invalid-lap flag | False |

Additional representative rows, covering a straight and a corner, with units and selection criteria: (To be added: a table of actual data samples)

### Figures and Quality Checks to Complete

The following analyses are planned but have not yet been completed.

| Artifact | Evidence to add |
| --- | --- |
| Track trajectory | (To be added: an x–z trajectory plot, travel direction, coverage, and interpretation) |
| Speed distribution or speed by track progress | (To be added: a plot with verified units and key observations) |
| Sampling-interval distribution | (To be added: a plot and summary showing the frequency of delays) |
| Section coverage | (To be added: sample counts by track section and an interpretation of coverage bias) |
| Data-quality table | (To be added: results for missing values, duplicates, NaNs, constant columns, non-monotonic timestamps, and timestamp alignment across files) |

In continuous control, what matters is how well the data covers different states and actions, rather than class balance as in classification. A single lap can't tell us how robust a policy is to different driving styles, starting conditions, or failures, and thousands of consecutive rows from that lap are not thousands of independent attempts.

### Failure Analysis and Infrastructure Checks

For A, we will look at the size of each reward component and check whether the agent gives up progress to avoid penalties. For B, we will compare behavior across track sections. For C, we will group repeated failures into types such as running wide at corner entry, steering oscillation, and getting stuck.

We will log infrastructure problems separately: failed resets, communication errors, and invalid observations. Before comparing any training runs, we will check coordinate consistency, normalization at zero speed, CPU/GPU handling, saving and loading, and control timing.

Common fixes and verification results: (To be added: a list of fixes, verification results, and the commit containing them)

Relevant code: [environment](https://github.com/Jurredr/ACRL/blob/master/standalone/sac/ac_environment.py), [path generation](https://github.com/Jurredr/ACRL/blob/master/track_data/path.py), and [networks](https://github.com/Jurredr/ACRL/blob/master/standalone/sac/core.py).

## 4. Evaluation Plan

### Metrics Aligned with the Main Goal

| Role | Measure |
| --- | --- |
| Primary if efficiency is selected | Environment steps to reach the preset lap-time and completion-rate target |
| Primary if efficacy is selected | Median valid lap time after a fixed training budget |
| Reliability requirement | Valid-lap completion rate and its change from the baseline |
| Learning diagnostics | Rates of reaching one sector, two sectors, and a full lap; progress before failure |
| Failure diagnostics | Number and duration of off-track events, and where failures keep happening |
| Control behavior | Size and frequency of steering changes |
| Computational cost | Gradient updates and wall-clock training time |

We will always report lap times alongside completion rates, so a model that sets a few fast laps but fails most attempts doesn't look better than it is.

### Outcome Definitions

| Outcome | Definition to finalize |
| --- | --- |
| Valid completion | A real timed lap, confirmed by the finish line or lap count and the validity flag; (To be added: rules for timing the start and finish and determining lap validity) |
| Driving failure | (To be added: failure conditions and thresholds for stopping, collisions, wrong-way driving, and other cases) |
| Off-track event | (To be added: used telemetry, duration thresholds, and rules for grouping consecutive events) |
| Timeout | (To be added: maximum steps and maximum wall-clock duration) |
| Infrastructure error | (To be added: detection, exclusion, and rerun rules, with error counts reported separately) |

Reaching nearly 100% progress is not enough to count as a valid lap. We will also separate warm-up or post-reset driving from the timed evaluation lap.

### Training, Validation, and Test Protocol

Training, validation (for tuning), and final testing will use separate sessions. We won't use test results to pick rewards, hyperparameters, or checkpoints, and evaluation data will never go into the training replay buffer.

| Protocol item | Planned value |
| --- | --- |
| Training seeds | We aim for at least 3 independent training seeds per condition; (To be added: the final seed list and number of runs) |
| Evaluation interval | (To be added: evaluation frequency in environment steps) |
| Validation episodes per checkpoint | (To be added: number of episodes) |
| Final test episodes per trained model | (To be added: number of episodes) |
| Evaluation actions | Deterministic policy actions with learning turned off |
| Initial conditions | (To be added: fixed and varied conditions, and the distinction between validation and test conditions) |
| Checkpoint selection | (To be added: a final-checkpoint or validation-based selection rule) |
| Efficiency target confirmation | (To be added: confirmation rules, including any requirement to sustain the target across consecutive evaluations) |

For efficiency, we will run validation evaluations on a fixed checkpoint schedule and then check the selected policy on held-out test runs. Because we only evaluate at checkpoints, we will report the schedule's resolution instead of claiming to know the exact step where the agent improved.

If a run never hits the target, we will report it as **not reached within budget**, rather than dropping it or counting the budget limit as its time. If a run completes no valid laps, its lap time will be listed as **N/A** alongside its completion and progress metrics.

### Controlled Comparisons and Reporting

Every condition will use the same interaction budget, update schedule, shared fixes, and evaluation setup. We will also record control intervals, since the same number of steps can mean a different amount of driving if timing varies. Any timing differences between conditions will be reported as a limitation.

- **A:** Change only the chosen penalty weight.
- **B:** Keep the architecture fixed. If we normalize inputs, compute the statistics from training data and apply them before masking. Keep preprocessing frozen during evaluation.
- **C:** Build the failure hypothesis from validation data, then test the fix on separate evaluation runs.

When reward functions differ across conditions, we will compare shared driving outcomes rather than raw cumulative rewards. We will report results for each training seed along with the spread across seeds and examples of typical failures. Running many evaluation episodes on one trained model is not a replacement for training with multiple seeds.

Result tables, uncertainty summaries, and failure examples: (To be added: actual experimental results and analysis)

## 5. Planned Neural Approach for Check-In 2

We will build on the existing [SAC implementation](https://github.com/Jurredr/ACRL/blob/master/standalone/sac/sac.py), keeping the model small to limit compute costs and changing one factor at a time to make the comparison easier to interpret.

| Component | Initial plan |
| --- | --- |
| Input | Default 10-dimensional state: progress, speed, 3D position, invalid-lap flag, lap count, previous progress, heading error, and centerline distance |
| Policy network | MLP with two 256-unit hidden layers and a squashed Gaussian action distribution |
| Critics | Two Q-networks taking state and action as input, each with two 256-unit hidden layers |
| Output | Two continuous actions: combined throttle/brake and steering |
| Training | SAC from scratch with a replay buffer and target networks |
| Preprocessing | (To be added: whether normalization is used and how its statistics are collected and frozen) |
| Initial intervention | (To be added: the selected experiment and exact intervention values) |
| Hyperparameters | (To be added: a configuration file specifying fixed learning rate, batch size, entropy coefficient, update schedule, and other settings) |

Architecture source: [core.py](https://github.com/Jurredr/ACRL/blob/master/standalone/sac/core.py).

For Check-In 2, we aim to have a working baseline and one comparison condition trained on a small, equal budget, with real learning curves or a documented account of how the agent fails. A non-neural probe is optional, and we don't plan to use a larger architecture.
