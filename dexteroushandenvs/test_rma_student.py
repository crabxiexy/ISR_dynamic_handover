
import os
import sys
import numpy as np
import time

# Manually handle custom argument for Adaptation Checkpoint before importing config
adaptation_checkpoint = None
if "--adaptation_checkpoint" in sys.argv:
    try:
        idx = sys.argv.index("--adaptation_checkpoint")
        adaptation_checkpoint = sys.argv[idx+1]
        # Remove from argv so get_args doesn't crash on unknown arg
        del sys.argv[idx:idx+2]
    except IndexError:
        print("Error: --adaptation_checkpoint requires a path argument")
        sys.exit(1)

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task
from tasks.allegro_hand_dynamic_handover_student import AllegroHandDynamicHandoverStudent
from algorithms.marl.runner import Runner
from tasks.hand_base.multi_vec_task_allegro import MultiVecTaskPythonAllegro

import torch

def test_rma():
    set_np_formatting()
    args = get_args()
    
    # Force mappo if not set
    if args.algo not in ["mappo", "happo", "hatrpo", "maddpg", "ippo"]:
        args.algo = "mappo"

    # Load Config
    cfg, cfg_train, logdir = load_cfg(args)
    
    # Overrides for Student Test
    cfg["env"]["use_adaptation"] = True
    args.task_type = "MultiAgent"
    cfg["is_test"] = True # Force test mode
    
    # Inject seed
    cfg["seed"] = cfg_train.get("seed", -1)
    cfg["env"]["seed"] = cfg["seed"]

    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    
    device = args.sim_device if args.use_gpu_pipeline else 'cpu'

    # Instantiate Environment
    agent_index = [[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]]
    
    task = AllegroHandDynamicHandoverStudent(
        cfg=cfg, 
        sim_params=sim_params, 
        physics_engine=args.physics_engine, 
        device_type=args.device, 
        device_id=args.device_id, 
        headless=args.headless, # Should be False for visual rollout
        agent_index=agent_index, 
        is_multi_agent=True
    )
    
    env = MultiVecTaskPythonAllegro(task, args.rl_device)
    
    # Setup Runner (loads Teacher Policy)
    config = cfg_train
    config["n_rollout_threads"] = env.num_envs
    config["n_eval_rollout_threads"] = env.num_envs
    config["device"] = device
    
    print(f"Loading Teacher Policy from: {args.model_dir}")
    marl_runner = Runner(vec_env=env, config=config, model_dir=args.model_dir)

    # Load Adaptation Module
    if adaptation_checkpoint:
        print(f"Loading Adaptation Module from: {adaptation_checkpoint}")
        task.load_adaptation_module(adaptation_checkpoint)
    else:
        print("WARNING: No adaptation checkpoint provided! Using random weights for adaptation.")
        
    # Set Phase to Student (Use Estimated Extrinsics)
    print("Setting RMA Phase to STUDENT (Inference)")
    task.set_rma_phase("student")
    
    # Evaluation Loop
    marl_runner.warmup()
    
    obs, share_obs, _ = env.reset()
    
    # Initialize RNN states
    rnn_states = torch.zeros(env.num_envs, marl_runner.num_agents, marl_runner.recurrent_N, marl_runner.hidden_size, device=device)
    rnn_states_critic = torch.zeros(env.num_envs, marl_runner.num_agents, *marl_runner.buffer[0].rnn_states_critic.shape[2:], device=device)
    masks = torch.ones(env.num_envs, marl_runner.num_agents, 1, device=device)

    print("Starting Rollout...")
    
    with torch.no_grad():
        while True:
            # Collect Actions (Deterministic for test)
            actions_collector = []
            
            # Helper to split obs/states for agents if needed, but marl_runner.policy.act handles it per agent?
            # Runner.collect logic is complex, let's look at `eval` logic in runner.py
            # Runner.eval resets everything. We want continuous loop.
            
            # We need to call policy for each agent.
            # obs: (N, num_agents, obs_dim) or list? 
            # MultiVecTaskPythonAllegro returns obs as (N, agents, dim) -> check step
            # step returns obs_all: (num_agents, num_envs, dim)?
            # Re-read MultiVecTaskPythonAllegro.step (Line 123):
            # obs_all = torch.transpose(torch.stack(sub_agent_obs), 1, 0) -> (num_envs, 2, dim)?
            # Wait, stack(sub_agent_obs) -> (2, num_envs, dim). transpose(1,0) -> (num_envs, 2, dim).
            # Yes.
            
            # runner.collect uses buffer access. Here we use direct tensors.
            
            step_actions = []
            
            for agent_id in range(marl_runner.num_agents):
                # marl_runner.trainer[agent_id].policy.actor(obs, rnn, masks, avail, deterministic)
                # obs[:, agent_id]
                
                agent_obs = obs[:, agent_id]
                agent_rnn = rnn_states[:, agent_id]
                agent_mask = masks[:, agent_id]
                
                action, _, new_rnn = marl_runner.policy[agent_id].actor(
                    agent_obs, agent_rnn, agent_mask, available_actions=None, deterministic=True
                )
                
                step_actions.append(action)
                rnn_states[:, agent_id] = new_rnn
                
            # Stack actions
            # step_actions list of (N, action_dim)
            # env expected: actions as tensor?
            # MultiVecTaskPythonAllegro.step expects `actions` as List/Tuple of tensors (one per agent) or stacked?
            # Line 91: a_hand_actions = actions[0] ... loop ... torch.hstack
            # It expects `actions` argument to be iterable of tensors, e.g. [action_agent1, action_agent2]
            
            obs, share_obs, rewards, dones, infos, _ = env.step(step_actions)
            
            # Handle resets for RNNs
            dones_env = torch.all(dones, dim=1)
            rnn_states[dones_env == True] = 0
            masks[dones_env == True] = 0
            masks[dones_env == False] = 1 # ? 
            # Masks logic: 0 if done, 1 otherwise
            masks = torch.ones(env.num_envs, marl_runner.num_agents, 1, device=device)
            masks[dones_env == True] = 0

            # Environment should render automatically if self.viewer exists (headless=False)
            
            # Optional: Sleep to slow down if too fast, but sim usually limits FPS
            # time.sleep(0.01)

if __name__ == "__main__":
    test_rma()
