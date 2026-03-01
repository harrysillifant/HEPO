import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import time

vec_env = make_vec_env("BipedalWalker-v3", n_envs=4, seed=42)

model = PPO("MlpPolicy", vec_env, verbose=1,
            tensorboard_log="./bipedal_tb_logs/")

# eval_env = make_vec_env("LunarLander-v3", n_envs=1)
# eval_callback = EvalCallback(
#     eval_env,
#     best_model_save_path="./ppo_logs",
#     log_path="./ppo_logs",
#     eval_freq=1000,
#     deterministic=True,
#     render=True,
# )

model.learn(
    total_timesteps=500_000,
    # callback=eval_callback,
    # log_interval=1,
    # tb_log_name="PPO",
    # reset_num_timesteps=True,
    progress_bar=True,
)

#
# render_env = gym.make("LunarLander-v3", render_mode="human")
#
# n_episodes = 5
# for ep in range(n_episodes):
#     obs, info = render_env.reset(seed=ep)
#     done = False
#     ep_reward = 0.0
#     while True:
#         action, _ = model.predict(obs, deterministic=True)
#         obs, reward, terminated, truncated, info = render_env.step(action)
#         ep_reward += float(reward)
#         # Many backends auto-render when render_mode="human"; call render() to be explicit
#         render_env.render()
#         if terminated or truncated:
#             print(f"Episode {ep + 1} reward: {ep_reward:.2f}")
#             break
#         # slow down so you can watch (optional)
#         time.sleep(1 / 60.0)
#
# render_env.close()
