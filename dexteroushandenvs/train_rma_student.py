
import os
import sys
import numpy as np
import time

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task
from tasks.allegro_hand_dynamic_handover_student import AllegroHandDynamicHandoverStudent
from algorithms.marl.runner import Runner
from tasks.hand_base.multi_vec_task_allegro import MultiVecTaskPythonAllegro

import torch
import torch.optim as optim

def train_rma():
    set_np_formatting()
    args = get_args()
    
    # Ensure algo is set to mappo if not specified, to trigger correct Runner logic
    if args.algo not in ["mappo", "happo", "hatrpo", "maddpg", "ippo"]:
        args.algo = "mappo"
        
    cfg, cfg_train, logdir = load_cfg(args)
    
    # Force specific task config overrides for Student
    cfg["env"]["use_adaptation"] = True
    args.task_type = "MultiAgent"
    
    # Set test mode logic for model loading (Runner conventions)
    if args.model_dir != "":
        cfg["is_test"] = True
    
    sim_params = parse_sim_params(args, cfg, cfg_train)
    seed = set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    
    device = args.sim_device if args.use_gpu_pipeline else 'cpu'
    
    # Instantiate Environment manually
    agent_index = [[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]] # Default
    
    task = AllegroHandDynamicHandoverStudent(
        cfg=cfg, 
        sim_params=sim_params, 
        physics_engine=args.physics_engine, 
        device_type=args.device, 
        device_id=args.device_id, 
        headless=args.headless,
        agent_index=agent_index, 
        is_multi_agent=True
    )
    
    # Wrap Task with MultiVecTaskPythonAllegro to comport with Runner expectation (vec_env.task)
    env = MultiVecTaskPythonAllegro(task, args.rl_device)
    
    # Setup Runner
    # config for runner needs strict keys
    config = cfg_train
    config["n_rollout_threads"] = env.num_envs
    config["n_eval_rollout_threads"] = env.num_envs
    config["device"] = device
    
    # We instantiate Runner which handles Model Loading (restore)
    print(f"Initializing Runner with model_dir: {args.model_dir}")
    env.task.cfg["seed"] = seed
    marl_runner = Runner(vec_env=env, config=config, model_dir=args.model_dir)
    
    # Adaptation Module Training Setup
    adaptation_module = task.adaptation_module
    optimizer = optim.Adam(adaptation_module.parameters(), lr=1e-3)
    
    num_epochs = 2000 
    save_interval = 50
    steps_per_epoch = 100 
    
    print("Starting RMA Phase 2 Training (Adaptation)...")
    
    # Set Phase to Teacher (Use GT for policy input)
    task.set_rma_phase("teacher") 
    
    # Initialize Runner buffers
    marl_runner.warmup()
    
    total_steps = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        for step in range(steps_per_epoch):
            # 1. Collect Data using Teacher Policy
            # We use marl_runner.collect(step=0) to get actions based on current buffer[0]
            # Since we don't advance the buffer (no PPO update), we reuse index 0.
            
            with torch.no_grad():
                values, actions, action_log_probs, rnn_states, rnn_states_critic = marl_runner.collect(step=0)
            
            # 2. Step Environment
            # This triggers compute_observations -> updates history -> runs adaptation (no_grad)
            obs, share_obs, rewards, dones, infos, _ = env.step(actions)
            
            # 3. Update Runner Buffer [0] manualy for next step
            # We must handle data placement and RNN state resets
            dones_env = torch.all(dones, dim=1)
            
            # Reset RNN states for done envs
            rnn_states[dones_env == True] = torch.zeros(
                (dones_env == True).sum(), marl_runner.num_agents, marl_runner.recurrent_N, marl_runner.hidden_size, device=device)
            rnn_states_critic[dones_env == True] = torch.zeros(
                (dones_env == True).sum(), marl_runner.num_agents, *marl_runner.buffer[0].rnn_states_critic.shape[2:], device=device)

            # Copy to buffer[0] for next iteration
            for agent_id in range(marl_runner.num_agents):
                marl_runner.buffer[agent_id].obs[0].copy_(obs[:, agent_id])
                marl_runner.buffer[agent_id].share_obs[0].copy_(share_obs[:, agent_id])
                marl_runner.buffer[agent_id].rnn_states[0].copy_(rnn_states[:, agent_id])
                marl_runner.buffer[agent_id].rnn_states_critic[0].copy_(rnn_states_critic[:, agent_id])
                
                # Update masks (used in act)
                marl_runner.buffer[agent_id].masks[0].copy_(torch.ones(marl_runner.n_rollout_threads, 1, device=device))
                marl_runner.buffer[agent_id].masks[0][dones_env == True] = 0

            # 4. Train Adaptation Module
            # Get data from Student Env
            # history: (N, 50, 108)
            # gt: (N, 16)
            
            pred_extrinsics = adaptation_module(task.obs_history.permute(0, 2, 1))
            target_extrinsics = task.gt_extrinsics
            
            # print("pred_extrinsics shape:", pred_extrinsics.shape)
            # print("target_extrinsics shape:", target_extrinsics.shape)
            loss = torch.mean((pred_extrinsics - target_extrinsics)**2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            total_steps += 1
            
        print(f"Epoch {epoch}: Avg Loss {epoch_loss / steps_per_epoch:.6f}")
        
        if epoch % save_interval == 0:
            save_path = os.path.join(logdir, f"adaptation_module_{epoch}.pt")
            torch.save(adaptation_module.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")

    # Final Save
    save_path = os.path.join(logdir, "adaptation_module_final.pt")
    torch.save(adaptation_module.state_dict(), save_path)
    print("Training Done")

if __name__ == "__main__":
    train_rma()
