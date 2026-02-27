### Issues
- When running on humanoid-bench, disable WANDB `export WANDB_MODE=offline` in terminal
- Problem with EvalCallback in run script, have disabled for now


general run command:
uv run hepo_hb.py --seed 0 --env_name h1hand-sit_simple-v0 --device cuda --num_envs 4 --max_steps 100000 --name gpu_test
