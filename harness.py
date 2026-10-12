import hashlib
import math
import numbers
import random
import time

from game_state import GameState

"""
harness.py

Training and evaluation routines shared by tester.py and the autograder, so that both measure your agent in exactly the
same way. You should not need to modify this file.

COMP3702 Assignment 3 "CrystalRover" Support Code

Last updated 09/10/2026
"""

EPISODE_TRIALS = 1000           # trials used to evaluate the final greedy policy
CHECKPOINT_COUNT = 25           # learning speed checkpoints spread across the training step budget
CHECKPOINT_TRIALS = 50          # trials per checkpoint evaluation
GENERALISATION_TRIALS = 100     # trials per generalisation start position

# The learning threshold sits this fraction of the way from the reward min target to the reward max target, i.e. it
# is the average reward that would earn three quarters of the episode reward marks.
LEARNED_REWARD_FRACTION = 0.75

MAX_FEATURES = 256

INITIALISE_TIME_LIMIT = 30      # seconds a *_initialise method may spend precomputing (this time is not marked)

PLAN_TYPES = ['ql', 'fq']
PLAN_NAMES = {'ql': 'Tabular Q-learning', 'fq': 'Feature-based Q-learning'}


class InvalidActionSelected(Exception):
    """
    Raised when a select_action method returns something that is not an element of GameEnv.ACTIONS.
    """

    def __init__(self, action):
        super().__init__(f'select_action returned {action!r}, which is not an element of GameEnv.ACTIONS')
        self.action = action


class TrainingRuleBroken(Exception):
    """
    Raised when a solver breaks a training rule the harness enforces, so the phase cannot be scored.
    """


class TrainingResult:
    """
    Summary of one training run.
    """

    def __init__(self):
        self.steps_used = 0
        self.episodes = 0
        self.training_time = 0.0
        self.training_reward = 0.0
        self.learned_at_steps = None
        self.checkpoint_history = []    # list of (training steps used, average checkpoint reward)
        self.ended_without_progress = False

    def get_time_per_step(self):
        if self.steps_used == 0:
            return 0.0
        return self.training_time / self.steps_used


def stable_hash(x):
    return hashlib.md5(str(x).encode('utf-8')).hexdigest()


def state_stable_hash(s: GameState):
    return stable_hash(str((s.row, s.col, s.crystal_status)))


def get_phase_value(env, plan_type, name):
    """
    Read a per-algorithm testcase value, e.g. get_phase_value(env, 'ql', 'reward_min_tgt') -> env.ql_reward_min_tgt.
    """
    return getattr(env, plan_type + '_' + name)


def get_solver_method(solver, plan_type, name):
    """
    Look up a per-algorithm solver method, e.g. get_solver_method(solver, 'fq', 'select_action').
    """
    return getattr(solver, plan_type + '_' + name)


def get_learning_threshold(env, plan_type):
    reward_min_tgt = get_phase_value(env, plan_type, 'reward_min_tgt')
    reward_max_tgt = get_phase_value(env, plan_type, 'reward_max_tgt')
    return reward_min_tgt + LEARNED_REWARD_FRACTION * (reward_max_tgt - reward_min_tgt)


def seed_solver_randomness(seed_value):
    """
    Seed the global random generators that solver code may use (e.g. for exploration), so that training runs are
    reproducible.
    """
    random.seed(seed_value)
    try:
        import numpy
    except ImportError:
        return
    numpy.random.seed(int(stable_hash(seed_value)[:8], 16))


# === Evaluation =======================================================================================================

def get_step_seed(control_env, trial, state, visit_count):
    # de-randomisation: the outcome depends only on the trial, the state and how often it has been visited
    return (str(control_env.episode_seed) + str(trial) + state_stable_hash(state) + stable_hash(visit_count))


def run_evaluation_episode(control_env, select_action, start_state, trial, reward_floor):
    """
    Run one episode of the given policy in the control environment.
    :return: (total episode reward, True if the level was solved)
    """
    state = start_state
    episode_reward = 0.0
    visit_count = {state: 1}
    for _ in range(control_env.max_episode_steps):
        if control_env.is_terminal(state) or episode_reward <= reward_floor:
            break
        action = select_action(state)
        if action not in control_env.ACTIONS:
            raise InvalidActionSelected(action)
        control_env.seed(get_step_seed(control_env, trial, state, visit_count[state]))
        state, reward, _ = control_env.perform_action(state, action)
        visit_count[state] = visit_count.get(state, 0) + 1
        episode_reward += reward
    return episode_reward, control_env.is_solved(state)


def evaluate_policy(control_env, select_action, start_state, trials, reward_floor, first_trial=0):
    """
    Evaluate the given policy over a number of seeded trials.
    :return: (average episode reward, number of trials in which the level was solved)
    """
    total_reward = 0.0
    solved_count = 0
    for trial in range(first_trial, first_trial + trials):
        episode_reward, solved = run_evaluation_episode(control_env, select_action, start_state, trial, reward_floor)
        total_reward += episode_reward
        if solved:
            solved_count += 1
    return total_reward / trials, solved_count


def evaluate_final_policy(control_env, select_action, plan_type):
    reward_floor = get_phase_value(control_env, plan_type, 'reward_min_tgt')
    return evaluate_policy(control_env, select_action, control_env.get_init_state(), EPISODE_TRIALS, reward_floor)


def evaluate_generalisation(control_env, select_action):
    """
    Evaluate the policy from each generalisation start position (with no crystals collected).
    :return: average episode reward over all start positions
    """
    no_crystals = tuple(0 for _ in control_env.crystal_positions)
    average_rewards = []
    for index, (row, col) in enumerate(control_env.generalisation_start_positions):
        start_state = GameState(row, col, no_crystals)
        average_reward, _ = evaluate_policy(control_env, select_action, start_state, GENERALISATION_TRIALS,
                                            control_env.fq_generalisation_min_tgt,
                                            first_trial=index * GENERALISATION_TRIALS)
        average_rewards.append(average_reward)
    return sum(average_rewards) / len(average_rewards)


# === Training =========================================================================================================

class CheckpointTracker:
    """
    Evaluates the greedy policy at regular intervals during training, and records the number of training steps used
    when its average reward first reaches the learning threshold.
    """

    def __init__(self, control_env, plan_type, step_budget):
        self.control_env = control_env
        self.threshold = get_learning_threshold(control_env, plan_type)
        self.reward_floor = get_phase_value(control_env, plan_type, 'reward_min_tgt')
        self.interval = step_budget / CHECKPOINT_COUNT
        self.next_checkpoint = self.interval
        self.learned_at_steps = None
        self.history = []

    def update(self, steps_used, select_action):
        if self.learned_at_steps is not None or steps_used < self.next_checkpoint:
            return
        self.evaluate(steps_used, select_action)
        self.next_checkpoint = (math.floor(steps_used / self.interval) + 1) * self.interval

    def finish(self, steps_used, select_action):
        """
        Evaluate once more at the end of training, unless the threshold was already reached or this point was already
        evaluated.
        """
        if self.learned_at_steps is not None or steps_used == 0:
            return
        if len(self.history) > 0 and self.history[-1][0] == steps_used:
            return
        self.evaluate(steps_used, select_action)

    def evaluate(self, steps_used, select_action):
        average_reward, _ = evaluate_policy(self.control_env, select_action, self.control_env.get_init_state(),
                                            CHECKPOINT_TRIALS, self.reward_floor)
        self.history.append((steps_used, average_reward))
        if average_reward >= self.threshold:
            self.learned_at_steps = steps_used


def call_within_budget(method, budget_exception):
    """
    Call a training method.
    :return: False if the training budget ran out during the call, True otherwise
    """
    try:
        method()
    except budget_exception:
        return False
    return True


def initialise_solver(training_env, solver, plan_type, budget_exception):
    """
    Call the solver's initialise method. It is not timed, so it may precompute, but it must not train and must finish
    within INITIALISE_TIME_LIMIT seconds.
    """
    start_time = time.time()
    call_within_budget(get_solver_method(solver, plan_type, 'initialise'), budget_exception)
    elapsed = time.time() - start_time
    if training_env.training_steps_used > 0:
        raise TrainingRuleBroken(f'{plan_type}_initialise called perform_action, but training must happen in '
                                 f'{plan_type}_train_episode.')
    if elapsed > INITIALISE_TIME_LIMIT:
        raise TrainingRuleBroken(f'{plan_type}_initialise took {elapsed:.1f} seconds, but precomputation must finish '
                                 f'within {INITIALISE_TIME_LIMIT} seconds.')


def train_agent(training_env, control_env, solver, plan_type, budget_exception):
    """
    Initialise the solver for the given plan type, then call its train_episode method until the training budget set on
    training_env is exhausted, checking learning speed at regular checkpoints.
    :param budget_exception: the TrainingBudgetExhausted class of the module training_env was created from
    :return: TrainingResult
    """
    train_episode = get_solver_method(solver, plan_type, 'train_episode')
    select_action = get_solver_method(solver, plan_type, 'select_action')
    tracker = CheckpointTracker(control_env, plan_type, training_env.training_step_budget)
    result = TrainingResult()

    initialise_solver(training_env, solver, plan_type, budget_exception)
    budget_left = True
    while budget_left:
        steps_before = training_env.training_steps_used
        start_time = time.time()
        budget_left = call_within_budget(train_episode, budget_exception)
        result.training_time += time.time() - start_time
        result.episodes += 1
        if budget_left and training_env.training_steps_used == steps_before:
            result.ended_without_progress = True
            break
        tracker.update(training_env.training_steps_used, select_action)

    tracker.finish(training_env.training_steps_used, select_action)
    result.steps_used = training_env.training_steps_used
    result.training_reward = training_env.training_reward_total
    result.learned_at_steps = tracker.learned_at_steps
    result.checkpoint_history = tracker.history
    return result


# === Feature checks ===================================================================================================

def get_feature_check_pairs(control_env):
    """
    (state, action) pairs used to check feature vectors: the initial state and every generalisation start state.
    """
    no_crystals = tuple(0 for _ in control_env.crystal_positions)
    states = [control_env.get_init_state()]
    for row, col in control_env.generalisation_start_positions:
        states.append(GameState(row, col, no_crystals))

    pairs = []
    for state in states:
        pairs += [(state, action) for action in control_env.get_valid_actions(state)]
    return pairs


def is_finite_number(value):
    return isinstance(value, numbers.Real) and math.isfinite(value)


def describe_feature_problem(features, expected_length):
    """
    :return: a description of what is wrong with the feature vector, or None if it is valid
    """
    if not hasattr(features, '__len__'):
        return f'fq_get_features returned {type(features).__name__}, expected a list of numbers'
    if len(features) == 0 or len(features) > MAX_FEATURES:
        return f'fq_get_features returned {len(features)} features, expected between 1 and {MAX_FEATURES}'
    if expected_length is not None and len(features) != expected_length:
        return f'fq_get_features returned vectors of different lengths ({expected_length} and {len(features)})'
    if not all(is_finite_number(value) for value in features):
        return 'fq_get_features returned a value that is not a finite number'
    return None


def check_feature_vectors(control_env, get_features):
    """
    Check that get_features returns a fixed-length vector of at most MAX_FEATURES finite numbers.
    :return: (number of features, None) if valid, otherwise (None, description of the problem)
    """
    feature_count = None
    for state, action in get_feature_check_pairs(control_env):
        features = get_features(state, action)
        problem = describe_feature_problem(features, feature_count)
        if problem is not None:
            return None, problem
        feature_count = len(features)
    return feature_count, None
