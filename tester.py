import sys
import time
import os

from game_env import GameEnv, TrainingBudgetExhausted
from harness import PLAN_TYPES, PLAN_NAMES, MAX_FEATURES, InvalidActionSelected, TrainingRuleBroken, \
    get_phase_value, get_solver_method, get_learning_threshold, get_step_seed, seed_solver_randomness, train_agent, \
    evaluate_final_policy, evaluate_generalisation, check_feature_vectors
from solution import Solver

"""
Tester script.

Use this script to debug and/or evaluate your solution. You may modify this file if desired.

The training and evaluation routines live in harness.py, which the autograder uses as well, so the numbers reported
here are computed the same way as the ones you are marked on.

COMP3702 Assignment 3 "CrystalRover" Support Code

Last updated 07/10/2026
"""

VISUALISE_TIME_PER_STEP = 0.7


def print_usage():
    print("Usage: python tester.py [plan_type] [testcase_file] [-v (optional)]")
    print("    plan_type = 'ql' (tabular Q-learning) or 'fq' (feature-based Q-learning)")
    print("    testcase_file = filename of a valid testcase file (e.g. L1.txt)")
    print("    if -v is specified, one episode of the trained policy will be visualised")


def parse_arguments(arglist):
    """
    :return: (plan_type, testcase_file, visualise), or None if the arguments are invalid
    """
    if len(arglist) != 2 and len(arglist) != 3:
        return None
    if arglist[0] not in PLAN_TYPES:
        print("/!\\ ERROR: Invalid plan_type given")
        return None
    if len(arglist) == 3 and arglist[2] != '-v':
        print(f"/!\\ ERROR: Invalid option given: {arglist[2]}")
        return None
    return arglist[0], arglist[1], len(arglist) == 3


def format_training_report(control_env, plan_type, result):
    step_budget = get_phase_value(control_env, plan_type, 'training_step_budget')
    reward_budget = get_phase_value(control_env, plan_type, 'training_reward_budget')
    msg = f'Training steps used: {result.steps_used} of {step_budget} in {result.episodes} episodes    ' \
          f'(cumulative training reward: {round(result.training_reward, 1)}, reward budget: {reward_budget})\n'
    if result.ended_without_progress:
        msg += 'Training stopped early: a train_episode call did not call perform_action.\n'

    threshold = round(get_learning_threshold(control_env, plan_type), 1)
    steps_max_tgt = get_phase_value(control_env, plan_type, 'learning_steps_max_tgt')
    if result.learned_at_steps is None:
        msg += f'Learning speed: the greedy policy never reached the learning threshold (average reward ' \
               f'{threshold})    (training steps max target: {steps_max_tgt})\n'
    else:
        msg += f'Learning speed: the greedy policy reached the learning threshold (average reward {threshold}) ' \
               f'after {result.learned_at_steps} training steps    (training steps max target: {steps_max_tgt})\n'

    time_max_tgt = get_phase_value(control_env, plan_type, 'time_max_tgt')
    msg += f'Average time per training step: {round(result.get_time_per_step(), 6)}    ' \
           f'(average time max target: {time_max_tgt})\n'
    return msg


def format_evaluation_report(control_env, plan_type, average_reward, solved_count):
    if solved_count > 0:
        msg = 'Level completed!\n'
    else:
        msg = 'Level not completed in any evaluation trial.\n'
    reward_max_tgt = get_phase_value(control_env, plan_type, 'reward_max_tgt')
    msg += f'Average episode reward: {round(average_reward, 1)}    (reward max target: {reward_max_tgt})\n'
    return msg


def format_feature_report(control_env, solver, select_action):
    feature_count, problem = check_feature_vectors(control_env, solver.fq_get_features)
    if problem is not None:
        return f'Feature vector check failed: {problem}\n'
    msg = f'Feature vector length: {feature_count}    (at most {MAX_FEATURES})\n'
    generalisation_reward = evaluate_generalisation(control_env, select_action)
    msg += f'Generalisation: average episode reward from {len(control_env.generalisation_start_positions)} ' \
           f'start positions: {round(generalisation_reward, 1)}    ' \
           f'(generalisation max target: {control_env.fq_generalisation_max_tgt})\n'
    return msg


def show_state(gui, control_env, state):
    if gui is not None:
        gui.update_state(state)
        return
    control_env.render(state)
    time.sleep(VISUALISE_TIME_PER_STEP)


def visualise_episode(game_env, control_env, select_action):
    """
    Show one episode of the trained greedy policy (trial 0 of the evaluation).
    """
    try:
        from gui import Viewer
        gui = Viewer(game_env)
    except ModuleNotFoundError:
        gui = None

    state = control_env.get_init_state()
    visit_count = {state: 1}
    show_state(gui, control_env, state)
    for _ in range(control_env.max_episode_steps):
        if control_env.is_terminal(state):
            break
        action = select_action(state)
        control_env.seed(get_step_seed(control_env, 0, state, visit_count[state]))
        state, reward, _ = control_env.perform_action(state, action)
        visit_count[state] = visit_count.get(state, 0) + 1
        print(f'Selected: {action} | Received a reward value of {reward}')
        show_state(gui, control_env, state)


def run_test(plan_type, testcase_file, visualise):
    game_env = GameEnv(testcase_file)
    control_env = GameEnv(testcase_file)

    # the budget is set before the solver is created, so that every perform_action call counts towards it
    game_env.set_training_budget(get_phase_value(game_env, plan_type, 'training_step_budget'),
                                 get_phase_value(game_env, plan_type, 'training_reward_budget'))
    seed_solver_randomness(control_env.episode_seed)
    solver = Solver(game_env)
    result = train_agent(game_env, control_env, solver, plan_type, TrainingBudgetExhausted)

    select_action = get_solver_method(solver, plan_type, 'select_action')
    average_reward, solved_count = evaluate_final_policy(control_env, select_action, plan_type)

    msg = f'===== Testcase {testcase_file.split("/")[-1].split(".")[0]} {PLAN_NAMES[plan_type]} =====\n'
    msg += format_training_report(control_env, plan_type, result)
    msg += format_evaluation_report(control_env, plan_type, average_reward, solved_count)
    if plan_type == 'fq':
        msg += format_feature_report(control_env, solver, select_action)
    print(msg)

    if visualise:
        visualise_episode(game_env, control_env, select_action)


def main(arglist):
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    arguments = parse_arguments(arglist)
    if arguments is None:
        print_usage()
        return

    plan_type = arguments[0]
    try:
        run_test(*arguments)
    except InvalidActionSelected as e:
        print(f'/!\\ ERROR: your {plan_type}_{e}')
    except TrainingRuleBroken as e:
        print(f'/!\\ ERROR: your {e}')


if __name__ == '__main__':
    main(sys.argv[1:])
