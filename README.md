# DSAN-6600-Project
Semester Project for DSAN 6600 Deep Learning and Neural Networks

This semester project uses the Soft Actor-Critic (SAC) agent implemented in [ACRL](https://github.com/Jurredr/ACRL) to study how one design choice affects driving behavior and failure. We will keep the scope narrow enough to run repeatable experiments within our available time and compute.

## 1. Research Goals and Scope

Our main goal is to improve ACRL's driving performance in one of two ways:

- **Learning efficiency:** reduce the number of environment interactions needed to reach a target lap time.
- **Driving efficacy:** achieve a shorter lap time than the baseline within the same training budget.

We will choose one as the primary objective and report the other as a secondary outcome. The initial experiments will use one car, one track, and fixed driving conditions.

### Main Quest: Learn Faster or Drive Faster

| Objective | Research question | Main measure |
| --- | --- | --- |
| **Efficiency** | Can the agent reach a target lap time with fewer training interactions? | Environment steps required to reach the target |
| **Efficacy** | Can the agent achieve a shorter lap time with the same training budget? | Median valid lap time during evaluation |

For this project, “iterations” will primarily mean **environment steps**: interactions in which the agent takes an action and receives a new observation. We will also report gradient updates and wall-clock training time, since an approach could use fewer driving steps while requiring more computation.

The baseline will be our reproducible run of ACRL with the common implementation fixes needed for valid experiments. We will not assume that an unverified published lap time is a reliable reference.

For efficiency, we will set a target lap time after a baseline pilot and before comparing proposed changes. We will evaluate at fixed training intervals and define the target as reached when the agent meets both:

- A specified median valid lap time
- A minimum valid-lap completion rate

For efficacy, we will compare frozen models after the same training budget. A faster lap will not count as an overall improvement if it comes with an unacceptable drop in valid-lap completion.

### Sub-Quests: Changes That Could Support the Main Goal
TBD

### What Counts as a Successful Result?

Our performance objective is a measurable improvement in efficiency or efficacy, supported by results across independent training seeds.

However, a proposed change may fail to improve either measure. In that case, the experiment can still contribute a useful finding if it:

- Uses a fair comparison
- Reports uncertainty and failure cases
- Shows which hypothesis the evidence supports or challenges
- Avoids claiming an improvement that the results do not establish

We will distinguish **achieving the performance objective** from **conducting a rigorous experiment**.

### Core Semester Scope

The main study will use:

- One car and one track
- Fixed driving conditions and control settings
- One primary objective: efficiency or efficacy
- One initial intervention
- A common training budget and evaluation protocol
- Multiple independent training seeds where feasible

Early experiments may use progress before failure or section-reaching rate to diagnose learning. These are supporting measures; they do not replace lap time as the main performance outcome.

If the baseline cannot complete valid laps reliably, the first milestone will be establishing a usable baseline. Until then, we cannot make a defensible claim about reaching a target lap time faster or beating the baseline lap time.

### Beyond Scope: Transfer to Another Track or Car

If the main experiment is complete and resources remain, we will investigate whether the approach can extend beyond the original car–track combination.

Our preferred first extension is **a different track with the same car**. We expect this may be easier than changing cars because the vehicle's dynamics and control interface remain more consistent. This is a working hypothesis, not an established result: a new track still introduces different geometry, braking points, and potentially different data requirements.

We will distinguish two transfer settings:

| Setting | Question | Evaluation |
| --- | --- | --- |
| **Zero-shot transfer** | Can the trained policy drive on a new track without additional learning? | Valid-lap completion and lap time with frozen model weights |
| **Fine-tuning** | Does starting from the trained policy make learning a new track more efficient? | Steps to a shared target compared with training from scratch on that track |

If the model requires a reference path or track-relative features, we will document how those inputs are obtained for the new track. Zero-shot transfer means no additional model training; it does not necessarily mean no track preparation.

A different car would be a further extension. Changes in acceleration, braking, grip, and steering response may require additional adaptation.

Transfer will remain a stretch goal so that it does not reduce the quality of the main comparison.

## 2. Data Access and Documentation

We need records of both the driving situation and the model's actions.

| Data | Purpose |
| --- | --- |
| Public ACRL CSV files | Initial checks of coordinates, recording intervals, and track paths |
| Logs from our own training runs | Comparisons of learning and outcomes for A and B |
| Evaluation logs from frozen models | Failure analysis for C and final evaluation for all options |

### Existing Data

The [public `track_data` directory](https://github.com/Jurredr/ACRL/tree/master/track_data) contains 15,681 recorded time points and 1,000 path points. Several CSVs describe different aspects of the same driving record, so their row counts should not be added together as independent samples.

These files do not provide complete state–action–reward–next-state transitions. They also lack the driver's steering and pedal inputs.

To access the files:

1. Open the [ACRL repository](https://github.com/Jurredr/ACRL).
2. Select **Code → Download ZIP**, or clone the repository.
3. Inspect the files in `track_data/`.
4. For new collection, follow the upstream [setup instructions](https://github.com/Jurredr/ACRL#readme) and verify the connection between the game and the external training program.

### New Data Collection

Each new record will include:

- Current state, executed action, reward, and next state
- Termination and truncation flags, with the reason an episode ended
- Timestamps and actual control intervals
- Session identifier, random seed, car, track, and experiment settings

A short pilot will establish whether collection works and how quickly training runs. We will then assign the same training-step budget to each condition.

Documentation will include the amount collected, field definitions and units, collection settings, and any excluded records with reasons. We will also record the upstream commit and our code version so the experiments can be reproduced.

### Usage and Licensing

The upstream ACRL repository includes a [GPL-3.0 license](https://github.com/Jurredr/ACRL/blob/master/LICENSE). We will document that license and separately state the sharing conditions for our own collected data.

### Pretrained Checkpoints

A pretrained model could be useful, but the initial review did not identify a ready-to-use ACRL checkpoint. Our plan therefore assumes short training runs of our own. If we obtain an external checkpoint, we will check its source, training conditions, and input/output compatibility before using it.

## 3. Data Audit and Failure Analysis

### Preliminary Findings

The initial audit of the public CSVs found:

| Item | Observation |
| --- | --- |
| Recorded time points | 15,681 |
| Recording duration | Approximately 523.3 seconds |
| Median recording interval | 33 ms |
| Recording interval range | 2–220 ms |
| Invalid-lap flag | `invalid=False` in every row |
| Action labels | No steering, throttle, or brake values |

The invalid flag indicates that the lap was not marked invalid. It does **not**, by itself, establish that the car stayed within the track at every moment. These records also provide limited evidence about failure frequencies or a trained model's driving ability.

Sources: [lap records](https://github.com/Jurredr/ACRL/blob/master/track_data/lap.csv) and [vehicle records](https://github.com/Jurredr/ACRL/blob/master/track_data/car.csv).

### Analysis by Research Direction
TBD

### Checks Before Experimentation
TBD

## 4. Evaluation Plan
TBD

### Metrics
TBD

### Training, Validation, and Testing
TBD

### Comparison Rules
TBD


## 5. Initial Neural Approach

We will start with [ACRL's MLP-based SAC implementation](https://github.com/Jurredr/ACRL/blob/master/standalone/sac/sac.py). Changes will be guided by what is needed to test the research question. Larger models or new input types can be considered if the initial experiments establish a reason to use them.

## Next Check-In
TBD







## Initial Brainstorming

Problem framing + scope

Efficient or Effective? (Maybe both)
Training an AI model to drive and set faster laptime possible or find a fast laptime with lower iteration.
Success Criteria: our laptime and/or number of iteration
Feasible semester scope: TBD
Why this problem matters to your group?
Video games are now more relying on AI, esp Sim genre, goal is AI setting competitive laptime

Dataset access + documentation

Dataset chosen and accessible
https://github.com/Jurredr/ACRL
Source, size, license/usage notes
Public Github repo of a complete Deep Reinforcement Learning model, not large for initial, originally developed by undergrad student from dutch Uni
Download or access instructions
Available public through github instruction of model use public gitbub repo, the game is 20$ on Steam

Data audit / EDA

Representative samples and summary stats
Class balance / key distributions as relevant
Obvious artifacts, biases, or likely failure modes

Evaluation plan

Metrics appropriate to the task
we need to measure laptime and number of iteration
laptime can be measured in comparison to either manual or built-in AI
Iteration (ofc lowest one we count)
The model as is will complete the task with x iteration, we want to lower that x!
Train/validation/test (or CV) plan


Initial direction

What neural approach you expect to try at Check-in 2

Any early non-neural probe is welcome but not required yet

Async video (~5 min)
