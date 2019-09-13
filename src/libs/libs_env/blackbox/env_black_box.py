import libs.libs_env.env as libs_env
import libs.libs_env.blackbox.black_box


class EnvBlackBox():

    def __init__(self, seed = 0):

        self.seed = seed

        self.black_box = libs.libs_env.blackbox.black_box.BlackBox(self.seed)

        self.width  = 1
        self.height = 1
        self.depth  = self.black_box.get_features_count()

        self.actions_count = self.black_box.get_actions_count()

    def print_info(self):
        print("BlackBox ENV")
        print("seed ", self.seed)
        print("features count ", self.get_features_count())
        print("actions count ", self.get_actions_count())
        print("\n\n")

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_depth(self):
        return self.depth

    def get_features_count(self):
        return self.black_box.get_features_count()

    def get_size(self):
        return self.black_box.get_features_count()

    def get_actions_count(self):
        return self.black_box.get_actions_count()

    def get_observation(self):
        return self.black_box.get_observation()

    def get_score(self):
        return self.black_box.get_score()

    def reset_score(self):
        self.black_box.reset_score()

    def do_action(self, action):
        self.black_box.do_action(action)

    def is_done(self):
        return self.black_box.is_new_game()

    def get_reward(self):
        return self.black_box.get_reward()

    def get_move(self):
        return self.black_box.get_iterations()

    def get_iterations(self):
        return self.black_box.get_iterations()

    def reset(self):
        self.black_box.init_new_game()
