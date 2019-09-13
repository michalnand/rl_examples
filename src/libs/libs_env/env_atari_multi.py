import libs.libs_env.env as env



import numpy
import time
import random



class EnvAtariMulti(env.Env):

    def __init__(self, envs, size = 32, game_change_iterations = 2048):
        self.envs = envs

        env.Env.__init__(self)

        self.game_change_iterations = game_change_iterations



        self.width  = self.envs[0].get_width()
        self.height = self.envs[0].get_height()
        self.depth  = self.envs[0].get_depth()
        self.time   = self.envs[0].get_time()

        self.active_env = -1
        self.__change_active_env()

        self.actions_count  = self.__get_actions_count()

        #init games
        self.reset()

        self.window_name = "ATARI - multi games"

    def get_observation(self):
        return self.envs[self.active_env].get_observation()

    def get_score(self):
        return self.envs[self.active_env].get_score()

    def reset_score(self):
        self.reward = 0.0
        self.score  = 0.0
        self.move   = 0
        for i in range(0, len(self.envs)):
            self.envs[i].reset_score()

    def reset(self):
        for i in range(0, len(self.envs)):
            self.envs[i].reset()

    def is_done(self):
        return self.envs[self.active_env].is_done()


    def _print(self):
        print("move=", self.get_move(), "  score=", self.get_score(), " game=", self.get_games_count())

    def do_action(self, action):

        action_ = action%self.envs[self.active_env].get_actions_count()

        self.envs[self.active_env].do_action(action_)

        self.reward = self.envs[self.active_env].get_reward()

        self.next_move()

        if self.get_move()%self.game_change_iterations == 0:
            self.__change_active_env()

    def render(self):
        self.envs[self.active_env].render()

    def get_envs_iterations(self):
        result = []
        for i in range(0, len(self.envs)):
            result.append(self.envs[i].get_move())

        return result

    def get_envs_games_count(self):
        result = []
        for i in range(0, len(self.envs)):
            result.append(self.envs[i].get_games_count())

        return result

    def get_envs_score(self):
        result = []
        for i in range(0, len(self.envs)):
            result.append(self.envs[i].get_score())

        return result

    def get_score(self):
        score = self.get_envs_score()
        sum = 0.0
        for i in range(0, len(score)):
            sum+= score[i]
        return sum

    def get_games_count(self):
        games_count = self.envs[0].get_games_count()

        for i in range(0, len(self.envs)):
            if self.envs[i].get_games_count() < games_count:
                games_count = self.envs[i].get_games_count()

        return games_count

    def get_active_env_id(self):
        return self.active_env

    def get_envs_count(self):
        return len(self.envs)

    def get_iterations(self):
        return self.move


    def set_heat_map(self, heat_map):
        self.envs[self.active_env].set_heat_map(heat_map)

    def __get_actions_count(self):
        actions_count = 0

        for i in range(0, len(self.envs)):
            if self.envs[i].get_actions_count() > actions_count:
                actions_count = self.envs[i].get_actions_count()

        return actions_count

    def __change_active_env(self):
        if len(self.envs) == 1:
            self.active_env = 0
        else:
            new_env = random.randint(0, len(self.envs) - 1)

            while new_env == self.get_active_env_id():
                new_env = random.randint(0, len(self.envs) - 1)

            self.active_env = new_env
