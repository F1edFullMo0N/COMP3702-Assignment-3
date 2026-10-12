# COMP3702 Assignment 3 Support Code

This is the support code for COMP3702 2026 Assignment 3, based on the CrystalRover game. The environment works as in
Assignment 2, but your agent now has to learn how to play it from experience, using tabular Q-learning and
Q-learning with features (linear function approximation).

The movement probabilities (drift, double moves and boost distance), the action costs (including the effect of the
storm) and the collision and lava penalties are now unknown: each testcase draws its own values from its seed, and the
environment keeps them private. Your agent has to discover their effects by interacting with the environment.

## Files

### game_env.py

This file defines the `GameEnv` class and contains the environment dynamics used by the assignment.

It stores the following key data:
- `n_rows`, `n_cols`
- `init_row`, `init_col`
- `crystal_positions`
- `launch_positions`
- `lava_positions`
- `gamma`, `max_episode_steps`
- `generalisation_start_positions`
- `ACTIONS`
- the training budget and score targets for each algorithm, e.g. `ql_training_step_budget`, `fq_reward_max_tgt`

The hidden parameters described above are private, and the ranges they are drawn from are listed as class constants
(e.g. `DRIFT_PROB_RANGE`).

Important methods:

~~~~~
__init__(filename)
~~~~~
Parses the testcase file and initialises the environment.

~~~~~
get_init_state()
~~~~~
Returns a GameState object (see below) representing the initial state of the level.

~~~~~
get_valid_actions(state)
~~~~~
Returns the actions that are valid on the rover's current tile: only jumps inside a crater, and only walks and boosts
everywhere else.

~~~~~
perform_action(state, action)
~~~~~
Applies the action to a state and returns the resulting state, the reward and an error message (`None` if the action
was performed normally). The action is stochastic: random drift and double-action noise may modify the intended move.
An action that is not valid on the current tile leaves the state unchanged but still incurs its action cost. Terminal
states are absorbing: acting in one returns the same state and zero reward.

~~~~~
is_solved(state), is_game_over(state), is_terminal(state)
~~~~~
`is_solved` returns `True` when the rover is on a launch tile and has collected at least `min_samples` crystals,
`is_game_over` returns `True` if the rover is on a lava tile, and `is_terminal` returns `True` in either case.

~~~~~
set_training_budget(step_budget, reward_budget), is_training_budget_exhausted()
~~~~~
The tester and autograder set a training budget before training starts. See "Training rules" below.

~~~~~
render(state)
~~~~~
Prints the current environment state to the terminal.

### game_state.py

This file defines the `GameState` object used throughout the solver and tester.

A `GameState` stores:
- `row`
- `col`
- `crystal_status` as a tuple of 0/1 values, where 1 means that crystal has been collected and 0 means it remains.

`GameState` objects can be compared with `==` and used as dictionary keys.

### solution.py

This is the template in which you implement your solver.

You must implement the following method stubs, which will be invoked by the tester and autograder:

- `__init__(game_env)`
- `ql_initialise()`, `ql_train_episode()`, `ql_select_action(state)` - tabular Q-learning
- `fq_initialise()`, `fq_get_features(state, action)`, `fq_train_episode()`, `fq_select_action(state)` -
  Q-learning with features, where Q(s, a) is the dot product of a learned weight vector with `fq_get_features(s, a)`

`fq_get_features` must return a list (or 1-D numpy array) of at most 256 numbers, with the same length for every
(state, action) pair. The `*_select_action` methods are used to evaluate your agent, so they should return the greedy
action for your learned values, without exploring or learning.

Implementing a 'main' method is not required.

To ensure compatibility with the autograder, please avoid using try-except blocks for Exception or OSError exception
types. Try-except blocks with concrete exception types other than OSError (e.g. try: ... except ValueError) are allowed.

Only `solution.py` (and any new files you create) should be submitted. Submitted copies of the support files
(`game_env.py`, `game_state.py`, `harness.py`, `tester.py`) are ignored by the autograder, so do not rely on changes
to them.

### tester.py

This script is used to debug and evaluate a candidate solver. It trains your agent with the testcase's training budget,
evaluates it, and reports the same measurements the autograder scores.

Usage:

```bash
python tester.py [ql|fq] [testcase_file] [-v]
```

`ql` runs tabular Q-learning and `fq` runs feature-based Q-learning. If `-v` is given, one episode of the trained
policy is visualised.

### harness.py

The training and evaluation routines shared by `tester.py` and the autograder. You should not need to modify this file.

### play_game.py

This script launches an interactive CrystalRover session.

Usage:

```bash
python play_game.py [input_filename] [--no-storm-particles]
```

The script loads a testcase, displays the board, and prompts the user for an action. Valid actions are the entries in
`GameEnv.ACTIONS`.

### gui.py

This file contains the visualiser used by the game and by the interactive play session. It renders the world, the
crystals, the rover, and storm effects in a graphical window.

### testcases/

This directory contains testcases L1 to L4. The autograder marks your agent on the same four maps but with different
episode seeds, so the hidden parameters and random outcomes differ, and different generalisation start positions, plus
a fifth level that is not released. Expect the autograder's numbers to be similar to, but not the same as, what
`tester.py` reports here: your agent has to learn each level, not rely on values tuned to these files.

## Training rules

The tester and autograder train your agent by calling `ql_initialise()` (or `fq_initialise()`) once, then calling
`ql_train_episode()` (or `fq_train_episode()`) repeatedly until the training budget is used up:

- The budget is a maximum number of `perform_action` calls and a lowest cumulative training reward, both given in the
  testcase. Once either runs out, `perform_action` raises `TrainingBudgetExhausted`, which ends training. Do not catch
  it. `game_env.training_steps_used`, `game_env.training_step_budget` and `game_env.training_reward_total` let your
  agent plan around the budget (e.g. to decay exploration). The reward budget is what our reference agent spends in
  training, so an agent that explores more carelessly (e.g. walks into lava more often) can run out before using all
  its training steps.
- During training, `perform_action` must be called on either the initial state (which starts a new episode) or the
  state returned by your previous `perform_action` call.
- `ql_initialise()` and `fq_initialise()` are not timed, so they are the place for precomputation, but they must not
  call `perform_action` and must finish within 30 seconds. Otherwise the phase scores zero (`tester.py` reports it as
  an error).
- Your agent must learn from experience. It may use the map (for example distances, or what lies in an action's
  direction) and the rewards it receives, but must not read the environment's private (hidden) parameters or call
  its private methods: how often actions go wrong, what they cost and how bad collisions and lava are have to be
  learned from what `perform_action` returns. Do not plan with a transition model (e.g. value iteration).

## How agents are evaluated

For each algorithm, the tester and autograder measure:

- **Learning speed**: every 1/25th of the step budget, the greedy policy is evaluated over 50 seeded episodes. This is
  the number of training steps used when its average reward first reaches the learning threshold, which lies three
  quarters of the way from the reward min target to the reward max target.
- **Time per training step**: the time spent in your `*_train_episode` methods divided by the training steps used
  (time in `*_initialise` is not counted).
- **Completion and episode reward**: the final greedy policy is evaluated over 1000 seeded episodes from the initial
  state.
- **Generalisation** (feature-based Q-learning only): the final greedy policy is also evaluated from each of the
  testcase's generalisation start positions, which your agent never starts a training episode from. The autograder
  uses start positions that are not in the released testcases.

Evaluation episodes end when the level is solved, the rover enters lava, `max_episode_steps` is reached, or the episode
reward falls to the reward min target.
