
from tasks.allegro_hand_dynamic_handover_teacher import AllegroHandDynamicHandoverTeacher
from algorithms.rma import AdaptationModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class AllegroHandDynamicHandoverStudent(AllegroHandDynamicHandoverTeacher):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless, agent_index, is_multi_agent)
        
        self.adaptation_history_len = self.cfg["env"].get("adaptation_history_len", 50)
        self.adaptation_input_dim = 16*4 + 44 # 108
        self.adaptation_output_dim = 16
        
        self.use_adaptation = self.cfg["env"].get("use_adaptation", True)
        self.device = self.sim_params.use_gpu_pipeline and "cuda" or "cpu"
        
        # History buffer: (N, HistoryLen, InputDim)
        self.obs_history = torch.zeros(self.num_envs, self.adaptation_history_len, self.adaptation_input_dim, device=self.device)
        
        # Adaptation Module
        self.adaptation_module = AdaptationModule(
            input_dim=self.adaptation_input_dim, 
            history_len=self.adaptation_history_len, 
            output_dim=self.adaptation_output_dim
        ).to(self.device)
        
        # If model dir is provided for adaptation module, load it
        # (This is for inference/test)
        if self.cfg.get("adaptation_model_path", "") != "":
             self.load_adaptation_module(self.cfg["adaptation_model_path"])
             
        self.gt_extrinsics = torch.zeros(self.num_envs, self.adaptation_output_dim, device=self.device)
        self.est_extrinsics = torch.zeros(self.num_envs, self.adaptation_output_dim, device=self.device)
        
        # Phase control: 'teacher' = Use GT obs (for data collection), 'student' = Use Est obs (for inference)
        self.rma_phase = "teacher" 

    def load_adaptation_module(self, path):
        if os.path.exists(path):
            self.adaptation_module.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Loaded Adaptation Module from {path}")
            self.adaptation_module.eval()
        else:
            print(f"Adaptation Module path {path} not found!")

    def set_rma_phase(self, phase):
        assert phase in ["teacher", "student"]
        self.rma_phase = phase
        print(f"RMA Phase switched to: {self.rma_phase}")

    def compute_observations(self):
        # 1. Compute Base Observations (Teacher's logic)
        # This fills self.obs_buf with GT params
        super().compute_observations()
        
        # 2. Extract GT Params for reference/training
        # Indices 263:279 are the privileged info (mass, inertia, etc.)
        self.gt_extrinsics = self.obs_buf[:, 263:279].clone()
        
        # 3. Update Adaptation History
        # Features: [DofPos1, DofVel1, DofPos2, DofVel2, Actions]
        # self.allegro_hand_dof_pos: (N, 16)
        # self.allegro_hand_dof_vel: (N, 16)
        # self.allegro_hand_another_dof_pos: (N, 16)
        # self.allegro_hand_another_dof_vel: (N, 16)
        # self.actions: (N, 44)
        
        current_step_features = torch.cat([
            self.allegro_hand_dof_pos, 
            self.allegro_hand_dof_vel, 
            self.allegro_hand_another_dof_pos, 
            self.allegro_hand_another_dof_vel,
            self.actions
        ], dim=-1) # (N, 108)
        
        # Shift history: remove oldest, add new
        # self.obs_history shape: (N, 50, 108)
        self.obs_history = torch.cat([
            self.obs_history[:, 1:, :], 
            current_step_features.unsqueeze(1)
        ], dim=1)
        
        # 4. Run Adaptation (Estimate Params)
        if self.use_adaptation:
            with torch.no_grad():
                # Input to module: (N, 108, 50)
                z_hat = self.adaptation_module(self.obs_history.permute(0, 2, 1))
            
            self.est_extrinsics = z_hat.clone()
            
            # 5. Modify Observations based on Phase
            if self.rma_phase == "student":
                # Replace GT in obs_buf with Estimate for the policy
                self.obs_buf[:, 263:279] = self.est_extrinsics
            
    def reset(self, env_ids, goal_env_ids):
        super().reset(env_ids, goal_env_ids)
        # Clear history for reset envs
        if len(env_ids) > 0:
            self.obs_history[env_ids] = 0.0
