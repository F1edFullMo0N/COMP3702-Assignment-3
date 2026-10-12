from game_state import GameState
import random

"""
game_env.py

COMP3702 Assignment 3 "CrystalRover" Support Code

Last updated 08/10/2026
"""


class TrainingBudgetExhausted(BaseException):
    """
    Raised by perform_action once the training budget set by set_training_budget() has been used up. The tester and
    autograder catch it to end training.

    Derives from BaseException (like KeyboardInterrupt) so that an `except Exception` block in solver code cannot
    swallow it.
    """
    pass


class GameEnv:
    """
    Instance of an Crystal Rover Game environment. Stores the dimensions of the environment, initial rover position,
    crater positions, rock positions, exit positions, crystal sample positions and the available actions.

    The movement probabilities (drift, double move and boost distance), the action costs (including the storm) and the
    collision and game over penalties are hidden: they are drawn from the testcase's seed, within the ranges below, and
    kept private. Your agent has to learn about them from what perform_action returns.
    """

    # input file symbols
    GROUND_TILE = ' '
    ROCK_TILE = 'R'
    CRATER_TILE = '*'
    CRYSTAL_TILE = 'C'
    LAUNCH_TILE = 'E'
    ROVER_TILE = 'P'
    LAVA_TILE = 'L'
    VALID_TILES = {GROUND_TILE, ROCK_TILE, CRATER_TILE, CRYSTAL_TILE, LAUNCH_TILE, ROVER_TILE, LAVA_TILE}

    # action symbols (i.e. output file symbols)
    WALK_LEFT = 'wl'
    WALK_RIGHT = 'wr'
    WALK_UP = 'wu'
    WALK_DOWN = 'wd'
    BOOST_LEFT = 'bl'
    BOOST_RIGHT = 'br'
    BOOST_UP = 'bu'
    BOOST_DOWN = 'bd'
    JUMP_LEFT = 'jl'
    JUMP_RIGHT = 'jr'
    JUMP_UP = 'ju'
    JUMP_DOWN = 'jd'

    ACTIONS = [
        WALK_LEFT, WALK_RIGHT, WALK_UP, WALK_DOWN,
        BOOST_LEFT, BOOST_RIGHT, BOOST_UP, BOOST_DOWN,
        JUMP_LEFT, JUMP_RIGHT, JUMP_UP, JUMP_DOWN,
    ]
    WALK_ACTIONS = {WALK_LEFT, WALK_RIGHT, WALK_UP, WALK_DOWN}
    BOOST_ACTIONS = {BOOST_LEFT, BOOST_RIGHT, BOOST_UP, BOOST_DOWN}
    JUMP_ACTIONS = {JUMP_LEFT, JUMP_RIGHT, JUMP_UP, JUMP_DOWN}

    # valid actions per tile type, in ACTIONS order so that iteration order is deterministic
    CRATER_VALID_ACTIONS = [JUMP_LEFT, JUMP_RIGHT, JUMP_UP, JUMP_DOWN]
    NON_CRATER_VALID_ACTIONS = [
        WALK_LEFT, WALK_RIGHT, WALK_UP, WALK_DOWN,
        BOOST_LEFT, BOOST_RIGHT, BOOST_UP, BOOST_DOWN,
    ]

    PERPENDICULAR_ACTIONS = {
        WALK_LEFT: [WALK_UP, WALK_DOWN],
        WALK_RIGHT: [WALK_UP, WALK_DOWN],
        WALK_UP: [WALK_LEFT, WALK_RIGHT],
        WALK_DOWN: [WALK_LEFT, WALK_RIGHT],
        BOOST_LEFT: [BOOST_UP, BOOST_DOWN],
        BOOST_RIGHT: [BOOST_UP, BOOST_DOWN],
        BOOST_UP: [BOOST_LEFT, BOOST_RIGHT],
        BOOST_DOWN: [BOOST_LEFT, BOOST_RIGHT],
        JUMP_LEFT: [JUMP_UP, JUMP_DOWN],
        JUMP_RIGHT: [JUMP_UP, JUMP_DOWN],
        JUMP_UP: [JUMP_LEFT, JUMP_RIGHT],
        JUMP_DOWN: [JUMP_LEFT, JUMP_RIGHT],
    }

    # ranges the hidden parameters are drawn from, per testcase
    DRIFT_PROB_RANGE = (0.1, 0.3)
    DOUBLE_PROB_RANGE = (0.1, 0.4)
    BOOST_WEIGHT_RANGES = [(0.05, 0.15), (0.2, 0.4), (0.2, 0.4), (0.1, 0.3), (0.05, 0.15)]  # boost distances 0 to 4
    BASE_COST_RANGES = {'walk': (0.8, 1.2), 'boost': (1.2, 1.8), 'jump': (1.7, 2.3)}
    STORM_DIRECTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN']
    STORM_COST_RANGE = (0.2, 0.6)
    COLLISION_PENALTY_RANGE = (0.5, 2.0)
    GAME_OVER_PENALTY_RANGE = (300.0, 700.0)

    # perform action return statuses
    SUCCESS = 0
    GAME_OVER = 2

    # number of non-comment lines before the grid data in a testcase file
    HEADER_LINES = 16

    def __init__(self, filename):
        """
        Parse the supplied testcase file and initialise the environment.
        """
        cleaned = self._read_testcase_lines(filename)
        assert len(cleaned) >= self.HEADER_LINES, '/!\\ ERROR: Invalid input file - expected header and grid data'

        try:
            self._parse_header(cleaned[:self.HEADER_LINES])
        except (ValueError, IndexError):
            assert False, '/!\\ ERROR: Invalid input file - malformed header values'

        # the simulator draws from its own generator, so randomness in solver code (e.g. exploration) never changes
        # the outcomes the environment produces
        self.random_generator = random.Random(self.episode_seed)

        grid_lines = cleaned[self.HEADER_LINES:self.HEADER_LINES + self.n_rows]
        self._parse_grid(grid_lines)
        self._check_generalisation_start_positions()

        self.n_crystals = len(self.crystal_positions)
        self.all_crystals_tuple = tuple([1 for _ in range(self.n_crystals)])
        self.__draw_hidden_parameters()

        # training budget, inactive until set_training_budget() is called
        self.training_budget_active = False
        self.training_step_budget = None
        self.training_reward_budget = None
        self.training_steps_used = 0
        self.training_reward_total = 0.0
        self.training_last_state = None

    @staticmethod
    def _read_testcase_lines(filename):
        try:
            with open(filename, 'r') as f:
                lines = [line.rstrip('\n') for line in f]
        except FileNotFoundError:
            assert False, '/!\\ ERROR: Testcase file not found'

        cleaned = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            cleaned.append(stripped)
        return cleaned

    def _parse_header(self, header):
        self.n_rows, self.n_cols = self._parse_values(header[0], int)
        self.num_rocket_jumps = header[1]  # infinite now :)
        self.min_samples = int(header[2])
        self.gamma = float(header[3])
        self.max_episode_steps = int(header[4])

        # tabular Q-learning budget and score targets
        self.ql_training_step_budget, self.ql_training_reward_budget = self._parse_budget(header[5])
        self.ql_reward_min_tgt, self.ql_reward_max_tgt = self._parse_values(header[6], float)
        self.ql_learning_steps_min_tgt, self.ql_learning_steps_max_tgt = self._parse_values(header[7], int)
        self.ql_time_min_tgt, self.ql_time_max_tgt = self._parse_values(header[8], float)

        # feature-based Q-learning budget and score targets
        self.fq_training_step_budget, self.fq_training_reward_budget = self._parse_budget(header[9])
        self.fq_reward_min_tgt, self.fq_reward_max_tgt = self._parse_values(header[10], float)
        self.fq_learning_steps_min_tgt, self.fq_learning_steps_max_tgt = self._parse_values(header[11], int)
        self.fq_time_min_tgt, self.fq_time_max_tgt = self._parse_values(header[12], float)
        self.fq_generalisation_min_tgt, self.fq_generalisation_max_tgt = self._parse_values(header[13], float)
        self.generalisation_start_positions = self._parse_positions(header[14])

        self.episode_seed = int(header[15])

    @staticmethod
    def _parse_values(line, value_type):
        return tuple(value_type(x) for x in line.split(','))

    @staticmethod
    def _parse_budget(line):
        step_budget, reward_budget = line.split(',')
        return int(step_budget), float(reward_budget)

    @staticmethod
    def _parse_positions(line):
        positions = []
        for entry in line.split(';'):
            row, col = entry.split(',')
            positions.append((int(row), int(col)))
        return positions

    def _parse_grid(self, grid_lines):
        assert len(grid_lines) == self.n_rows, '/!\\ ERROR: Invalid input file - incorrect number of map rows - expected {} but got {}'.format(self.n_rows, len(grid_lines))

        self.grid_data = []
        self.init_row, self.init_col = None, None
        self.crystal_positions = []
        self.launch_positions = []
        self.lava_positions = []

        for r, row in enumerate(grid_lines):
            self.grid_data.append(self._parse_grid_row(r, row))

        assert self.init_row is not None and self.init_col is not None, '/!\\ ERROR: Invalid input file - No player initial position'
        assert len(self.launch_positions) > 0, '/!\\ ERROR: Invalid input file - No launch position'

    def _parse_grid_row(self, r, row):
        chars = list(row)
        assert len(chars) == self.n_cols, '/!\\ ERROR: Invalid input file - incorrect map row length - expected {} but got {}'.format(self.n_cols, len(chars))
        for c, ch in enumerate(chars):
            if ch == self.ROVER_TILE:
                assert self.init_row is None and self.init_col is None, '/!\\ ERROR: Invalid input file - more than one initial player position'
                self.init_row, self.init_col = r, c
                chars[c] = self.GROUND_TILE
            elif ch == self.LAUNCH_TILE:
                self.launch_positions.append((r, c))
            elif ch == self.CRYSTAL_TILE:
                self.crystal_positions.append((r, c))
                chars[c] = self.GROUND_TILE
            elif ch == self.LAVA_TILE:
                self.lava_positions.append((r, c))
        return chars

    def _check_generalisation_start_positions(self):
        for row, col in self.generalisation_start_positions:
            assert 0 <= row < self.n_rows and 0 <= col < self.n_cols, \
                '/!\\ ERROR: Invalid input file - generalisation start position outside the map'
            is_plain_ground = self.grid_data[row][col] == self.GROUND_TILE and \
                (row, col) not in self.crystal_positions
            assert is_plain_ground, '/!\\ ERROR: Invalid input file - generalisation start position must be ground'

    def __draw_hidden_parameters(self):
        # drawn from a generator of their own, so that they do not depend on how the simulation is seeded
        parameter_random = random.Random(f'{self.episode_seed}-hidden-parameters')
        self.__random_drift_prob = parameter_random.uniform(*self.DRIFT_PROB_RANGE)
        self.__random_double_prob = parameter_random.uniform(*self.DOUBLE_PROB_RANGE)
        boost_weights = [parameter_random.uniform(*weight_range) for weight_range in self.BOOST_WEIGHT_RANGES]
        self.__boost_probabilities = tuple(weight / sum(boost_weights) for weight in boost_weights)
        self.__collision_penalty = parameter_random.uniform(*self.COLLISION_PENALTY_RANGE)
        self.__game_over_penalty = parameter_random.uniform(*self.GAME_OVER_PENALTY_RANGE)
        self.__storm_direction = parameter_random.choice(self.STORM_DIRECTIONS)
        self.__storm_directional_cost = parameter_random.uniform(*self.STORM_COST_RANGE)
        base_costs = {action_type: parameter_random.uniform(*cost_range)
                      for action_type, cost_range in self.BASE_COST_RANGES.items()}
        self.__action_costs = self.__build_action_costs(base_costs)

    def __build_action_costs(self, base_costs):
        action_costs = {}
        for action in self.ACTIONS:
            cost = base_costs[self.__get_action_type(action)]
            direction = self._action_direction(action)
            if direction == self.__storm_direction:
                cost -= self.__storm_directional_cost
            elif direction == self._opposite_direction(self.__storm_direction):
                cost += self.__storm_directional_cost
            action_costs[action] = cost
        return action_costs

    def __get_action_type(self, action):
        if action in self.WALK_ACTIONS:
            return 'walk'
        if action in self.BOOST_ACTIONS:
            return 'boost'
        return 'jump'

    def _action_direction(self, action):
        mapping = {
            self.WALK_LEFT: 'LEFT', self.BOOST_LEFT: 'LEFT', self.JUMP_LEFT: 'LEFT',
            self.WALK_RIGHT: 'RIGHT', self.BOOST_RIGHT: 'RIGHT', self.JUMP_RIGHT: 'RIGHT',
            self.WALK_UP: 'UP', self.BOOST_UP: 'UP', self.JUMP_UP: 'UP',
            self.WALK_DOWN: 'DOWN', self.BOOST_DOWN: 'DOWN', self.JUMP_DOWN: 'DOWN',
        }
        return mapping.get(action)

    def _opposite_direction(self, direction):
        return {
            'LEFT': 'RIGHT',
            'RIGHT': 'LEFT',
            'UP': 'DOWN',
            'DOWN': 'UP',
        }.get(direction)


    def get_init_state(self):
        """
        Get a state representation instance for the initial state.
        :return: initial state
        """
        state = GameState(self.init_row, self.init_col, tuple(0 for _ in self.crystal_positions))
        return state

    def get_valid_actions(self, state):
        """
        Get the actions that are valid on the rover's current tile: only jumps inside a crater, and only walks and
        boosts everywhere else.
        :param state: current GameState
        :return: list of valid actions (in GameEnv.ACTIONS order)
        """
        if self.grid_data[state.row][state.col] == self.CRATER_TILE:
            return list(self.CRATER_VALID_ACTIONS)
        return list(self.NON_CRATER_VALID_ACTIONS)

    def perform_action(self, state, action):
        """
        Perform the given action on the given state and return the resulting new state, the reward received and an
        error message (None if the action was performed normally).

        Terminal states (solved or game over) are absorbing: acting in one returns the same state and zero reward. An
        action that is not valid on the current tile (see get_valid_actions) leaves the state unchanged but still
        incurs its action cost.

        Once set_training_budget() has been called, every call counts against the training budget, and must be made
        on either the initial state (starting a new episode) or the state returned by the previous call.
        """
        self._begin_training_step(state)
        next_state, reward, err_msg = self._simulate_action(state, action)
        self._end_training_step(next_state, reward)
        return next_state, reward, err_msg

    def _simulate_action(self, state, action):
        if action not in self.ACTIONS:
            return state, 0.0, "Invalid action"
        if self.is_terminal(state):
            return state, 0.0, "Episode is over - state is terminal"
        if action not in self.get_valid_actions(state):
            return state, -1 * self.__action_costs[action], "Action is not valid on the current tile"

        # apply drift and double probs
        movements = self.__apply_action_noise(action)

        # apply dynamics based on movements
        new_state = state.deepcopy()
        total_reward = 0.0
        err_msg = None
        for m in movements:
            valid, err_msg, next_state, reward, terminal_state = self.__apply_dynamics(new_state, m)
            if not valid:
                continue
            new_state = next_state
            total_reward += reward
            if terminal_state:
                break
        return new_state, total_reward, err_msg

    def set_training_budget(self, step_budget, reward_budget):
        """
        Start enforcing a training budget. Called by the tester and autograder before training begins.

        Training ends (perform_action raises TrainingBudgetExhausted) once step_budget calls to perform_action have
        been made, or once the cumulative reward received during training falls to reward_budget or below.
        :param step_budget: maximum number of perform_action calls
        :param reward_budget: lowest cumulative training reward allowed (a negative number)
        """
        self.training_budget_active = True
        self.training_step_budget = step_budget
        self.training_reward_budget = reward_budget
        self.training_steps_used = 0
        self.training_reward_total = 0.0
        self.training_last_state = None

    def is_training_budget_exhausted(self):
        """
        Check if the training budget has been used up (always False when no budget has been set).
        :return: True if exhausted, False otherwise
        """
        if not self.training_budget_active:
            return False
        if self.training_steps_used >= self.training_step_budget:
            return True
        return self.training_reward_total <= self.training_reward_budget

    def _begin_training_step(self, state):
        if not self.training_budget_active:
            return
        if self.is_training_budget_exhausted():
            raise TrainingBudgetExhausted(
                f'Training budget exhausted after {self.training_steps_used} steps '
                f'(cumulative training reward {round(self.training_reward_total, 1)})')
        if state != self.get_init_state() and state != self.training_last_state:
            raise ValueError('/!\\ ERROR: During training, perform_action must be called on the initial state (to '
                             'start a new episode) or on the state returned by the previous perform_action call.')

    def _end_training_step(self, next_state, reward):
        if not self.training_budget_active:
            return
        self.training_steps_used += 1
        self.training_reward_total += reward
        self.training_last_state = next_state

    def seed(self, seed_value):
        """
        Re-seed the random generator that the simulator draws action outcomes from.
        :param seed_value: any value accepted by random.seed()
        """
        self.random_generator.seed(seed_value)

    def is_solved(self, state):
        """
        Check if the game has been solved (i.e. player at exit and minimum number of crystals collected)
        :param state: current GameState
        :return: True if solved, False otherwise
        """
        collected = sum(state.crystal_status)
        return (state.row, state.col) in self.launch_positions and collected >= self.min_samples

    def is_game_over(self, state):
        """
        The game is over if the player is in lava.
        :param state: current GameState
        :return: True if game is over, False otherwise
        """
        return (self.grid_data[state.row][state.col] == self.LAVA_TILE)

    def is_terminal(self, state):
        """
        Check if the episode has ended (level solved or game over).
        :param state: current GameState
        :return: True if terminal, False otherwise
        """
        return self.is_solved(state) or self.is_game_over(state)

    def render(self, state):
        """
        Render the map's current state to terminal.
        """
        for r in range(self.n_rows):
            line = ''
            for c in range(self.n_cols):
                if state.row == r and state.col == c:
                    line += 'P'
                elif (r, c) in self.launch_positions:
                    line += 'E'
                elif (r, c) in self.lava_positions:
                    line += 'L'
                elif (r, c) in self.crystal_positions and state.crystal_status[self.crystal_positions.index((r, c))] == 0:
                    line += 'C'
                else:
                    line += self.grid_data[r][c]
            print(line)
        print('\n' * 2)


    def __apply_action_noise(self, action):
        """
        Apply the action noise to the given action and return the resulting action.
        :param action: action string
        :return: resulting action string
        """
        movements = []

        drifted = self.random_generator.random() < self.__random_drift_prob
        doubled = self.random_generator.random() < self.__random_double_prob

        if drifted:
            # randomly drift perpendicular to the intended direction
            perpendiculars = self.PERPENDICULAR_ACTIONS[action]
            drift_choice = self.random_generator.choice(perpendiculars)
            movement = drift_choice
        else:
            movement = action

        movements.append(movement)
        if doubled:
            movements.append(movement)

        return movements

    def __apply_dynamics(self, state, action):
        """
        Apply the dynamics of the game to the given state and action and return the resulting state and reward.
        :param state: current GameState
        :param action: action string
        :return: action is valid (True/False), error message if invalid, next state, reward, state is terminal
        """

        if action not in self.ACTIONS:
            return False, "Invalid action", None, 0.0, None

        reward = -1 * self.__action_costs[action]
        next_row, next_col = state.row, state.col

        direction = self._action_direction(action)

        deltas = {
            'LEFT': (0, -1),
            'RIGHT': (0, 1),
            'UP': (-1, 0),
            'DOWN': (1, 0),
        }

        delta_row, delta_col = deltas[direction]

        if action in self.JUMP_ACTIONS:
            if self.grid_data[state.row][state.col] != self.CRATER_TILE:
                return False, "Cannot perform rocket jump", None, 0.0, None
            move_distance = 1
        elif action in self.WALK_ACTIONS:
            if self.grid_data[state.row][state.col] == self.CRATER_TILE:
                return False, "Cannot perform action: in a crater", None, 0.0, None
            move_distance = 1
        elif action in self.BOOST_ACTIONS:
            if self.grid_data[state.row][state.col] == self.CRATER_TILE:
                return False, "Cannot perform action: in a crater", None, 0.0, None
            # sample the boost distance based on the boost probabilities
            move_distance = self.random_generator.choices([0, 1, 2, 3, 4], weights=self.__boost_probabilities, k=1)[0]

        collision = False
        for _ in range(move_distance):
            candidate_row = next_row + delta_row
            candidate_col = next_col + delta_col
            if not (0 <= candidate_row < self.n_rows and 0 <= candidate_col < self.n_cols) \
                    or self.grid_data[candidate_row][candidate_col] == self.ROCK_TILE:
                reward -= self.__collision_penalty
                collision = True
                break

            next_row, next_col = candidate_row, candidate_col

            # fall into a crater
            if self.grid_data[next_row][next_col] == self.CRATER_TILE:
                break

            # fall into lava
            if self.grid_data[next_row][next_col] == self.LAVA_TILE:
                reward -= self.__game_over_penalty
                break

        crystal_status = state.crystal_status
        if (next_row, next_col) in self.crystal_positions:
            crystal_index = self.crystal_positions.index((next_row, next_col))
            if crystal_status[crystal_index] == 0:
                crystal_status = list(crystal_status)
                crystal_status[crystal_index] = 1
                crystal_status = tuple(crystal_status)

        next_state = GameState(next_row, next_col, crystal_status)
        if not collision and self.is_game_over(next_state) and \
                self.grid_data[next_row][next_col] != self.LAVA_TILE:
            reward -= self.__game_over_penalty

        return True, None, next_state, reward, self.is_game_over(next_state)
