
import os
import sys
import numpy as np
import torch
import torch.optim as optim
import time

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task
from tasks.allegro_hand_dynamic_handover_student import AllegroHandDynamicHandoverStudent
from algorithms.sarl.ppo.module import ActorCritic

def train_rma():
    set_np_formatting()
    args = get_args()
    cfg, cfg_train, logdir = load_cfg(args)
    
    # Force specific task config overrides for Student
    cfg["env"]["use_adaptation"] = True
    # Ensure numObservations matches the Teacher's size (901 or whatever)
    # The student class inherits it, so it should be fine.
    
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    
    device = args.sim_device if args.use_gpu_pipeline else 'cpu'
    
    # Instantiate Environment manually
    # We need to constructing agent_index same as parse_task usually does
    # For single agent, it's usually just defaults.
    agent_index = [[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]] # From teacher file default
    
    env = AllegroHandDynamicHandoverStudent(
        cfg=cfg, 
        sim_params=sim_params, 
        physics_engine=args.physics_engine, 
        device_type=args.device, 
        device_id=args.device_id, 
        headless=args.headless,
        agent_index=agent_index, 
        is_multi_agent=False
    )
    
    # Load Teacher Policy
    # We assume standard PPO ActorCritic
    # Check dimensions
    # obs_shape: (num_obs,)
    # states_shape: (num_states,)
    # actions_shape: (num_actions,)
    
    obs_shape = (env.num_obs,) # e.g. (901,)
    states_shape = (env.num_states,) # e.g. (215,) or (0,)
    actions_shape = (env.num_actions,) # 44
    
    print(f"Obs Shape: {obs_shape}, States Shape: {states_shape}, Actions Shape: {actions_shape}")
    
    # Model Config
    policy_cfg = cfg_train["policy"]
    
    actor_critic = ActorCritic(
        obs_shape=obs_shape,
        states_shape=states_shape,
        actions_shape=actions_shape,
        initial_std=0.0, # Not training policy
        model_cfg=policy_cfg,
        asymmetric=(env.num_states > 0)
    ).to(device)
    
    if args.model_dir:
        print(f"Loading Teacher Policy from {args.model_dir}")
        loaded_dict = torch.load(args.model_dir, map_location=device)
        actor_critic.load_state_dict(loaded_dict['model'])
        actor_critic.eval()
    else:
        print("WARNING: No teacher model provided! Use --model_dir path/to/model.pt")
    
    # Adaptation Module Training Setup
    adaptation_module = env.adaptation_module
    optimizer = optim.Adam(adaptation_module.parameters(), lr=1e-3)
    
    num_epochs = 1000 # Example
    save_interval = 50
    steps_per_epoch = 100 # Steps to collect before update?
    # Usually RMA collects a buffer and updates.
    # Here let's do online updates: Step Env -> Compute Loss -> Backprop
    
    # Or batch updates: Accumulate gradients over N steps?
    
    print("Starting Adaptation Training...")
    
    env.reset_buf[:] = 1 # Force reset first
    obs = env.reset() # This gives obs with ZERO history and ESTIMATE=0
    # Ideally we should warm up history with Teacher using GT 
    # But here we just start.
    
    total_steps = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        for step in range(steps_per_epoch):
            # 1. Get Action from Teacher Policy
            # obs contains current estimate.
            # We want policy to act based on estimate (Student Mode)
            # OR act based on GT (Teacher Mode) while training student?
            # Standard RMA: Data collection is done with TEACHER Policy (using GT).
            # So, we should temporarily revert env to use GT for the POLICY input?
            # BUT, we want to train the student to recover GT.
            # If we run with GT, the states visited are "good".
            # If we run with Estimate (untrained), the states might be "bad" (falling).
            # RMA Paper: "We collect data using the base policy \pi" (which uses z_t).
            # So the Environment should provide obs with *GT* to the policy.
            # BUT Adaptation Module trains on the history generated.
            
            # CURRENT IMPLEMENTATION of StudentEnv: 
            # compute_observations replaces obs_buf param slots with ESTIMATE.
            
            # WORKAROUND:
            # We can manually put GT back into obs before calling policy.
            # gt_extrinsics are stored in env.gt_extrinsics
            
            obs_for_policy = obs.clone()
            # Replace estimate with GT for stable data collection
            obs_for_policy[:, 263:279] = env.gt_extrinsics
            
            with torch.no_grad():
                actions = actor_critic.act_inference(obs_for_policy)
            
            # 2. Step Env
            # step() will call pre_physics, physics, post_physics
            # post_physics calls compute_observations -> updates history, runs adaptation -> updates obs
            obs, rewards, dones, infos = env.step(actions)
            
            # 3. Train Adaptation
            # env.obs_history (N, 50, 108)
            # env.gt_extrinsics (N, 16)
            # Adaptation forward pass
            
            # Note: compute_observations already ran adaptation forward (no_grad). 
            # We need to run it AGAIN with grad.
            
            pred_extrinsics = adaptation_module(env.obs_history.permute(0, 2, 1))
            target_extrinsics = env.gt_extrinsics
            
            loss = torch.mean((pred_extrinsics - target_extrinsics)**2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            total_steps += 1
            
            if total_steps % 1000 == 0:
                print(f"Step {total_steps}, Loss: {loss.item()}")
                
        print(f"Epoch {epoch}: Avg Loss {epoch_loss / steps_per_epoch}")
        
        if epoch % save_interval == 0:
            save_path = os.path.join(logdir, f"adaptation_module_{epoch}.pt")
            torch.save(adaptation_module.state_dict(), save_path)
            print(f"Saved model to {save_path}")

    # Final Save
    save_path = os.path.join(logdir, "adaptation_module_final.pt")
    torch.save(adaptation_module.state_dict(), save_path)
    print("Training Done")

if __name__ == "__main__":
    train_rma()
