import libs.libs_env.blackbox.black_box_match



match = libs.libs_env.blackbox.black_box_match.BlackBoxMatch("black_box_match_config.json")


match.run()
match.print_score()
match.save_score()
