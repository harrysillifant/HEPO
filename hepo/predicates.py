from typing import Callable


def LunarLander_predicate():
    def task_predicate(obs, reward, done, info):
        terminated = bool(info.get("terminal_observation") is not None)
        truncated = bool(info.get("TimeLimit.truncated"))
        if terminated and not truncated:
            if reward not in (-100.0, 100.0):
                print("WRONG")
                print("Info is ", info)
                print("Reward is ", reward)
        return terminated and not truncated

    return task_predicate


def BipedalWalker_predicate():
    def task_predicate(obs, reward, done, info):
        terminated = bool(info.get("terminal_observation") is not None)
        truncated = bool(info.get("TimeLimit.truncated"))
        if terminated and not truncated:
            if reward not in (-100.0, 300.0):
                print("WRONG")
                print("Info is ", info)
                print("Reward is ", reward)
        return terminated and not truncated

    return task_predicate
