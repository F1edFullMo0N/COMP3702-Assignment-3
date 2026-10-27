import sys
import time
import random
from game_env import GameEnv
from game_state import GameState
from collections import deque
"""
solution.py

This file is a template you should use to implement your solution.

You should implement each of the method stubs below. You may add additional methods and/or classes to this file if you
wish. You may also create additional source files and import to this file if you wish.

COMP3702 Assignment 3 "CrystalRover" Support Code

Last updated 16/10/2026
"""


class Solver:

    def __init__(self, game_env: GameEnv):
        self.game_env = game_env
        #
        # TODO: Define any class instance variables you require (e.g. a dictionary mapping (state, action) to Q-value,
        #  or the weights of your linear Q-function) here.
        #
        pass

    @staticmethod
    def testcases_to_attempt():
        """
        Return a list of testcase numbers you want your solution to be evaluated for.
        """
        # TODO: modify below if desired (e.g. disable larger testcases if you're having problems with RAM usage, etc)
        return [1, 2, 3, 4, 5]

    # === Training rules (apply to both algorithms) ====================================================================
    #
    # Before training starts, the tester and autograder give self.game_env a training budget: a maximum number of
    # perform_action calls, and a lowest cumulative training reward. Once either runs out, perform_action raises
    # TrainingBudgetExhausted, which ends training - do not catch it. self.game_env.training_steps_used,
    # self.game_env.training_step_budget and self.game_env.training_reward_total let you plan around the budget.
    #
    # During training, perform_action must be called on either the initial state (self.game_env.get_init_state(), which
    # starts a new episode) or the state returned by your previous perform_action call.
    #
    # Your agent must learn from experience. The movement probabilities, action costs and penalties are hidden: each
    # testcase draws its own from its seed, and the environment keeps them private. Your agent may use the map (e.g.
    # distances, or what lies in an action's direction) and the rewards it receives, but must not read the
    # environment's private parameters or call its private methods, or plan with a transition model (e.g. value
    # iteration).
    #
    # Only some actions are valid on each tile (self.game_env.get_valid_actions(state)). An invalid action leaves the
    # state unchanged and still costs its action cost.

    # === Tabular Q-learning ===========================================================================================

    def ql_initialise(self):
        """
        Initialise any variables required before the start of tabular Q-learning.
        """
        #
        # TODO: Implement any initialisation for tabular Q-learning (e.g. creating your Q-table) here. This method is
        #  not timed, but it must not call perform_action (training belongs in ql_train_episode) and must finish
        #  within 30 seconds, or the phase scores zero.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        
        self.q_table = {}
        self.alpha = 0.2
        self.epsilon = 0.3

    def ql_train_episode(self):
        """
        Run one episode of tabular Q-learning.
        """
        #
        # TODO: Implement a single training episode here: start from self.game_env.get_init_state() and repeatedly
        #  choose an action (e.g. epsilon-greedy), call self.game_env.perform_action and update your Q-values, until
        #  the episode ends (a terminal state, or a step limit such as self.game_env.max_episode_steps).
        #
        # The tester and autograder call this method repeatedly until the training budget is exhausted, and check how
        # well your greedy policy performs at regular checkpoints in between.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        state = self.game_env.get_init_state()
        episode_reward = 0.0

        # Decay epsilon based on training progress
        budget = self.game_env.training_step_budget
        if budget is None:
            epsilon = self.epsilon
        else:
            progress = min(
                1.0,
                self.game_env.training_steps_used / budget
            )
            epsilon = max(
                0.05,
                self.epsilon * (1.0 - progress)
            )

        for _ in range(self.game_env.max_episode_steps):

            # Stop if episode has ended
            if (self.game_env.is_terminal(state)
                    or episode_reward <= self.game_env.ql_reward_min_tgt):
                break

            # Epsilon-greedy action selection
            actions = self.game_env.get_valid_actions(state)

            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                action = max(
                    actions,
                    key=lambda a: self.q_table.get((state, a), 0.0)
                )

            # Perform action
            next_state, reward, _ = self.game_env.perform_action(
                state, action
            )

            # Q-learning update
            old_q = self.q_table.get((state, action), 0.0)

            if self.game_env.is_terminal(next_state):
                next_best_q = 0.0
            else:
                next_best_q = max(
                    self.q_table.get((next_state, a), 0.0)
                    for a in self.game_env.get_valid_actions(next_state)
                )

            target = reward + self.game_env.gamma * next_best_q

            self.q_table[(state, action)] = (
                old_q + self.alpha * (target - old_q)
            )

            # Move to next state
            state = next_state
            episode_reward += reward

    def ql_select_action(self, state: GameState):
        """
        Select the greedy action for the given state according to your learned Q-values.
        :param state: the current state
        :return: greedy action for the given state (element of GameEnv.ACTIONS)
        """
        #
        # TODO: Implement code to return the greedy action for the given state (based on your learned Q-values) here.
        #  This method is used to evaluate your agent (including during training), so it should not explore or update
        #  your Q-values.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        
        actions = self.game_env.get_valid_actions(state)

        return max(
            actions,
            key=lambda a: self.q_table.get((state, a), 0.0)
        )


    # === Feature-based Q-learning =====================================================================================

    def fq_initialise(self):
        """
        Initialise any variables required before the start of feature-based Q-learning.
        """
        #
        # TODO: Implement any initialisation for feature-based Q-learning (e.g. initial weights, or pre-computing
        #  information about the map that your features use) here. This method is not timed, so it is the place for
        #  precomputation, but it must not call perform_action (training belongs in fq_train_episode) and must finish
        #  within 30 seconds, or the phase scores zero.
        #
        # Your weights must start all 0 or all 1 (e.g. [0.0] * number of features) and be learned during training:
        #  starting from hand-picked weights scores zero for the phase.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        
        self.fq_alpha = 0.10
        self.fq_epsilon = 0.40
        # Labels are useful for report Q3(b).
        self.fq_feature_names = [
            "Bias", "Negative remaining-route distance", "Route progress",
            "Nominal lava exposure", "Perpendicular drift lava exposure",
            "Double movement lava exposure", "Rock collision exposure",
            "Crater landing", "Boost action", "Jump action",
            "Walk action", "Crystal acquired", "Finish route",
            "Left", "Right", "Up", "Down",
        ]
        self.fq_weights = [0.0] * len(self.fq_feature_names)
        self.fq_feature_cache = {}
        env = self.game_env
        self.fq_crystals = tuple(env.crystal_positions)
        self.fq_exits = tuple(env.launch_positions)
        self.fq_crystal_index = {xy:i for i,xy in enumerate(self.fq_crystals)}
        self.fq_grid = env.grid_data
        self.fq_lava = set(env.lava_positions)
        self.fq_rows, self.fq_cols = env.n_rows, env.n_cols
        self.fq_dirs = {'l': (0,-1), 'r': (0,1), 'u': (-1,0), 'd': (1,0)}
        self.fq_scale = float(max(env.n_cols + env.n_rows, 1))
        walkable = {(r,c) for r in range(env.n_rows) for c in range(env.n_cols)
                    if env.grid_data[r][c] != env.ROCK_TILE and (r,c) not in self.fq_lava}
        # Static map distance only: no learned/hidden transition probabilities.
        self.fq_dist = {}
        for source in walkable:
            seen = {source:0}
            queue=deque([source])
            while queue:
                r,c=queue.popleft()
                for dr,dc in self.fq_dirs.values():
                    neighbor=(r+dr,c+dc)
                    if neighbor in walkable and neighbor not in seen:
                        seen[neighbor]=seen[(r,c)]+1
                        queue.append(neighbor)
            self.fq_dist[source]=seen
        self.fq_route_cache = {}

    def fq_remaining_distance(self, pos, status):
        """Estimate shortest remaining route through enough crystals to a launch site."""
        key=(pos,status)
        if key in self.fq_route_cache:
            return self.fq_route_cache[key]
        collected=sum(status)
        need=max(0, self.game_env.min_samples - collected)
        if need == 0:
            ans=min((self.fq_dist.get(pos, {}).get(e, 999) for e in self.fq_exits),default=999)
        else:
            # Recursion only traverses a small *static map* collection task.
            ans=999
            ds=self.fq_dist.get(pos, {})
            for i,c in enumerate(self.fq_crystals):
                if status[i]:
                    continue
                d=ds.get(c,999)
                if d >= 999:continue
                new_status=status[:i]+(1,)+status[i+1:]
                ans=min(ans, d+self.fq_remaining_distance(c,new_status))
        self.fq_route_cache[key]=ans
        return ans

    def fq_simulate_geometry(self, state, action, length=None):
        """Local terrain-only look-ahead for constructing descriptive features.

        Not used for transition planning or estimating hidden movement probabilities.
        """
        if length is None:
            length=2 if action[0]=='b' else 1
        pos=(state.row,state.col)
        status=state.crystal_status
        dr,dc=self.fq_dirs[action[1]]
        hit_rock=0
        hit_lava=0
        for _ in range(length):
            new=(pos[0]+dr,pos[1]+dc)
            if (new[0] < 0 or new[0] >= self.fq_rows or new[1] < 0 or new[1] >= self.fq_cols
                    or self.fq_grid[new[0]][new[1]]==self.game_env.ROCK_TILE):
                hit_rock=1
                break
            pos=new
            if pos in self.fq_lava:
                hit_lava=1
                break
            if self.fq_grid[pos[0]][pos[1]]==self.game_env.CRATER_TILE:
                break
        acquired=0
        if not hit_lava and pos in self.fq_crystal_index:
            i=self.fq_crystal_index[pos]
            if not status[i]:
                acquired=1
                status=status[:i]+(1,)+status[i+1:]
        crater=1 if self.fq_grid[pos[0]][pos[1]] == self.game_env.CRATER_TILE else 0
        finish=1 if pos in self.fq_exits and sum(status)>=self.game_env.min_samples else 0
        return pos,status,hit_rock,hit_lava,crater,acquired,finish

    def fq_value(self, state: GameState, action):
        features = self.fq_get_features(state, action)
        return sum(
            w * f
            for w, f in zip(self.fq_weights, features)
        )

    def fq_get_features(self, state: GameState, action):
        """
        Return the feature vector of the given (state, action) pair, so that Q(s, a) = w . features(s, a).
        :param state: the current state
        :param action: an element of GameEnv.ACTIONS
        :return: a list (or 1-D numpy array) of at most 256 numbers, the same length for every (state, action) pair
        """
        #
        # TODO: Implement your feature design here. Features that describe the situation (e.g. distances, risks) rather
        #  than identify the exact state are what let your agent act well in states it never visited during training.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        
        key=(state,action)
        if key in self.fq_feature_cache:
            return self.fq_feature_cache[key]
        pos=(state.row,state.col)
        status=state.crystal_status
        new_pos,new_status,collide,lava,crater,acquired,finish = self.fq_simulate_geometry(state,action)
        d_before=self.fq_remaining_distance(pos,status)
        d_after=self.fq_remaining_distance(new_pos,new_status)
        # For boost, assess 0..4 tile outcomes as geometry indicators, NOT a probability model.
        if action[0]=='b':
            raw=[self.fq_simulate_geometry(state,action,length=i) for i in range(5)]
            nominal_lava=sum(x[3] for x in raw)/5.0
            collide=sum(x[2] for x in raw)/5.0
        else:
            nominal_lava=float(lava)
        # A perpendicular deviation can land in a hazard even for a safe intended move.
        drifts=[]
        for d in (('u','d') if action[1] in ('l','r') else ('l','r')):
            alternate=action[0]+d
            if alternate[0]=='b':
                raw=[self.fq_simulate_geometry(state,alternate,length=i) for i in range(5)]
                drifts.append(sum(x[3] for x in raw)/5.0)
            else:
                drifts.append(float(self.fq_simulate_geometry(state,alternate)[3]))
        drift_lava=sum(drifts)/2.0
        # Possible second movement from approximate landing position.
        dummy=GameState(new_pos[0],new_pos[1],new_status)
        doublerisk=0.0
        if not lava and not finish:
            doublerisk=float(self.fq_simulate_geometry(dummy, action)[3])
        dist_penalty=-min(d_after,100)/self.fq_scale
        progress=max(-4,min(4,d_before-d_after))/4.0
        hazard_scale=100.0
        features=[
            1.0, dist_penalty, progress,
            -hazard_scale*nominal_lava,
            -hazard_scale*drift_lava,
            -hazard_scale*doublerisk,
            -float(collide), -float(crater),
            float(action[0]=='b'), float(action[0]=='j'), float(action[0]=='w'),
            float(acquired), float(finish),
            float(action[1]=='l'),float(action[1]=='r'),
            float(action[1]=='u'),float(action[1]=='d')
        ]
        self.fq_feature_cache[key]=features
        return features

    def fq_train_episode(self):
        """
        Run one episode of feature-based Q-learning (Q-learning with linear function approximation).
        """
        #
        # TODO: Implement a single training episode here, as for ql_train_episode, but updating the weights of your
        #  linear Q-function instead of a Q-table.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        
        state=self.game_env.get_init_state()
        episode_reward=0.0
        env=self.game_env
        budget=env.training_step_budget
        progress=min(1.0,env.training_steps_used/budget) if budget else 0.0
        epsilon=max(0.05,self.fq_epsilon*(1.0-progress))
        for _ in range(env.max_episode_steps):
            if env.is_terminal(state) or episode_reward<=env.fq_reward_min_tgt:
                break
            actions=env.get_valid_actions(state)
            if random.random()<epsilon:
                action=random.choice(actions)
            else:
                action=self.fq_select_action(state)
            features=self.fq_get_features(state,action)
            old_q=sum(w*f for w,f in zip(self.fq_weights,features))
            next_state,reward,_=env.perform_action(state,action)
            episode_reward+=reward
            if env.is_terminal(next_state):
                target=reward
            else:
                next_actions=env.get_valid_actions(next_state)
                target=reward+env.gamma*max(self.fq_value(next_state,a) for a in next_actions)
            td_error=max(-750.0,min(750.0,target-old_q))
            norm=sum(f*f for f in features)
            if norm>0:
                step=self.fq_alpha*td_error/norm
                self.fq_weights=[w+step*f for w,f in zip(self.fq_weights,features)]
            state=next_state


    def fq_select_action(self, state: GameState):
        """
        Select the greedy action for the given state according to your learned weights.
        :param state: the current state
        :return: greedy action for the given state (element of GameEnv.ACTIONS)
        """
        #
        # TODO: Implement code to return the greedy action for the given state (based on your learned weights) here.
        #  This method is also evaluated from start positions your agent never trained from.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        
        actions=self.game_env.get_valid_actions(state)
        return max(actions,key=lambda a:self.fq_value(state,a))

    # === Helper Methods ===============================================================================================
    #
    #
    # TODO: Add any additional methods here
    #
    #
