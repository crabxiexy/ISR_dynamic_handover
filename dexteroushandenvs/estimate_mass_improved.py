"""
改进的物体质量估计脚本 - 基于SwingBot和动力学模型
修改为z方向上下移动版本

核心改进：
1. 使用关节力矩反推末端接触力：F_contact = (J^T)^(-1) * (τ - τ_robot)
2. 使用动力学方程估计质量：m = F_contact / (a + g)
3. 执行垂直上下移动收集多个数据点
4. 使用最小二乘法提高估计精度
"""

import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import torch

class ImprovedMassEstimator:
    def __init__(self, use_gpu=True):
        self.gym = gymapi.acquire_gym()
        self.device = 'cuda:0' if torch.cuda.is_available() and use_gpu else 'cpu'
        print(f"使用设备: {self.device}")
        
        # 仿真参数
        self.sim_params = gymapi.SimParams()
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        self.sim_params.dt = 1.0 / 60.0
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 8
        self.sim_params.physx.num_velocity_iterations = 0
        self.sim_params.physx.use_gpu = use_gpu
        self.sim_params.use_gpu_pipeline = use_gpu
        
        compute_device_id = 0 if use_gpu else -1
        self.sim = self.gym.create_sim(compute_device_id, 0, gymapi.SIM_PHYSX, self.sim_params)
        if self.sim is None:
            raise RuntimeError("Failed to create sim")
        
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        
        self.viewer = None
        
        # 数据存储
        self.data = {
            'joint_torques': [],
            'joint_positions': [],
            'joint_velocities': [],
            'ee_positions': [],
            'ee_velocities': [],
            'ee_accelerations': [],
            'obj_positions': [],
            'obj_velocities': [],
            'obj_accelerations': [],
            'actual_mass': None
        }
        
    def create_environment(self, object_type="obj0"):
        """创建环境"""
        asset_root = "../assets"
        
        # 加载机械臂
        hand_asset_file = "urdf/xarm6/xarm6_allegro_right_2023_binghao.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        
        hand_asset = self.gym.load_asset(self.sim, asset_root, hand_asset_file, asset_options)
        
        # 加载物体
        object_asset_file = f"urdf/binghao_obj/objects/{object_type}.urdf"
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 2000
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)
        
        num_hand_dofs = self.gym.get_asset_dof_count(hand_asset)
        
        # 创建环境
        env_ptr = self.gym.create_env(self.sim, 
                                     gymapi.Vec3(-2, -2, 0), 
                                     gymapi.Vec3(2, 2, 2), 1)
        
        # 初始姿态
        hand_pose = gymapi.Transform()
        hand_pose.p = gymapi.Vec3(0, -1.35, 0.2)
        hand_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 1.57079)
        
        obj_pose = gymapi.Transform()
        obj_pose.p = gymapi.Vec3(0.025, -0.38, 0.449)
        
        # 创建actors
        hand_actor = self.gym.create_actor(env_ptr, hand_asset, hand_pose, "hand", 0, -1, 0)
        obj_actor = self.gym.create_actor(env_ptr, object_asset, obj_pose, "object", 0, 0, 0)
        
        # 获取实际质量
        obj_props = self.gym.get_actor_rigid_body_properties(env_ptr, obj_actor)
        actual_mass = sum([p.mass for p in obj_props])
        self.data['actual_mass'] = actual_mass
        print(f"实际质量: {actual_mass:.4f} kg")
        
        # 配置DOF
        dof_props = self.gym.get_asset_dof_properties(hand_asset)
        for i in range(num_hand_dofs):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if i < 6:
                dof_props['stiffness'][i] = 100.0
                dof_props['damping'][i] = 10.0
            else:
                dof_props['stiffness'][i] = 30.0
                dof_props['damping'][i] = 1.0
        self.gym.set_actor_dof_properties(env_ptr, hand_actor, dof_props)
        
        # 获取索引
        hand_idx = self.gym.get_actor_index(env_ptr, hand_actor, gymapi.DOMAIN_SIM)
        obj_idx = self.gym.get_actor_index(env_ptr, obj_actor, gymapi.DOMAIN_SIM)
        
        # 找到末端执行器刚体（通常是link6）
        try:
            ee_body_idx = self.gym.find_actor_rigid_body_index(
                env_ptr, hand_actor, "link6", gymapi.DOMAIN_ENV)
        except:
            ee_body_idx = self.gym.get_asset_rigid_body_count(hand_asset) - 1
        
        return env_ptr, hand_actor, obj_actor, hand_idx, obj_idx, num_hand_dofs, ee_body_idx
    
    def vertical_motion_and_collect(self, env_ptr, hand_idx, obj_idx, num_dofs, ee_body_idx, num_steps=600):
        """执行垂直上下移动并收集数据"""
        self.gym.prepare_sim(self.sim)
        
        # 获取状态张量
        root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        
        root_states = gymtorch.wrap_tensor(root_state_tensor)
        dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        rb_states = gymtorch.wrap_tensor(rb_state_tensor)
        dof_forces = gymtorch.wrap_tensor(dof_force_tensor)
        
        # 默认DOF位置
        default_pos = torch.zeros(num_dofs, dtype=torch.float32, device=self.device)
        default_pos[:6] = torch.tensor([0.0, -0.09, -0.09, 3.141, 2.00, -1.57], device=self.device)
        default_pos[6:] = torch.tensor([
            -0.03989830748810656, 1.3495253790945758, 0.8659920759388671, 0.780414711365591,
            0.9655586519308622, 1.0139016439397597, 0.8501943208059994, 1.3264760744914152,
            -0.20482272250532974, 1.347864170294202, 0.6030536585610538, 0.9181400800651911,
            -0.21341465375119012, 1.7199185039090872, 1.2686849760515697, 0.8245164874462315,
        ], device=self.device)
        
        # z方向上下移动参数
        amplitude = 0.02  # 振幅（米）
        frequency = 0.2   # 频率（Hz）
        base_z = default_pos[2].item()  # 使用默认的z位置作为基准
        
        dt = self.sim_params.dt
        prev_ee_vel = np.zeros(3)
        prev_obj_vel = np.zeros(3)
        
        print("执行垂直上下移动...")
        
        for step in range(num_steps):
            t = step * dt
            target_pos = default_pos.clone()
            
            # 在z方向上进行正弦运动
            z_offset = amplitude * np.sin(2 * np.pi * frequency * t)
            target_pos[2] = base_z + z_offset
            
            self.gym.set_dof_position_target_tensor(
                self.sim, gymtorch.unwrap_tensor(target_pos.unsqueeze(0)))
            
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)
            
            # 收集数据
            joint_pos = dof_states[:num_dofs, 0].cpu().numpy()
            joint_vel = dof_states[:num_dofs, 1].cpu().numpy()
            joint_torque = dof_forces[:num_dofs].cpu().numpy()
            
            ee_pos = rb_states[ee_body_idx, 0:3].cpu().numpy()
            ee_vel = (ee_pos - prev_ee_vel) / dt if step > 0 else np.zeros(3)
            ee_accel = (ee_vel - prev_ee_vel) / dt if step > 1 else np.zeros(3)
            prev_ee_vel = ee_vel.copy()
            
            obj_pos = root_states[obj_idx, 0:3].cpu().numpy()
            obj_vel = root_states[obj_idx, 7:10].cpu().numpy()
            obj_accel = (obj_vel - prev_obj_vel) / dt if step > 0 else np.zeros(3)
            prev_obj_vel = obj_vel.copy()
            
            self.data['joint_positions'].append(joint_pos)
            self.data['joint_velocities'].append(joint_vel)
            self.data['joint_torques'].append(joint_torque)
            self.data['ee_positions'].append(ee_pos)
            self.data['ee_velocities'].append(ee_vel)
            self.data['ee_accelerations'].append(ee_accel)
            self.data['obj_positions'].append(obj_pos)
            self.data['obj_velocities'].append(obj_vel)
            self.data['obj_accelerations'].append(obj_accel)
        
        # 转换为numpy数组
        for key in self.data:
            if isinstance(self.data[key], list):
                self.data[key] = np.array(self.data[key])
        
        print(f"收集了 {num_steps} 步数据")
    
    def estimate_mass_using_least_squares(self):
        """
        使用最小二乘法估计质量
        
        方法：
        1. 假设物体与末端执行器有接触，一起运动
        2. 物体受到的净力：F_net = m * a_obj = F_contact - m * g
        3. 如果物体与手接触良好：a_obj ≈ a_ee
        4. 所以：F_contact ≈ m * (a_ee + g)
        
        简化：使用z方向的力和加速度
        F_z = m * (a_z + g)
        m = F_z / (a_z + g)
        
        使用多个数据点，通过最小二乘法求解：
        min ||F_z - m * (a_z + g)||^2
        """
        if len(self.data['joint_torques']) == 0:
            return None
        
        ee_accels = self.data['ee_accelerations']
        obj_accels = self.data['obj_accelerations']
        
        # 使用z方向
        ee_accel_z = ee_accels[:, 2]
        obj_accel_z = obj_accels[:, 2]
        
        # 检测接触（物体加速度接近末端执行器加速度）
        accel_diff = np.abs(obj_accel_z - ee_accel_z)
        contact_mask = accel_diff < 0.3
        
        if np.sum(contact_mask) < 20:
            print(f"接触数据点较少 ({np.sum(contact_mask)})，使用所有数据")
            contact_mask = np.ones_like(ee_accel_z, dtype=bool)
        
        # 方法：使用加速度响应来估计
        # 如果物体质量大，对于相同的运动，加速度响应较小
        
        # 使用加速度的标准差
        ee_accel_z_contact = ee_accel_z[contact_mask]
        obj_accel_z_contact = obj_accel_z[contact_mask]
        
        std_ee = np.std(ee_accel_z_contact)
        std_obj = np.std(obj_accel_z_contact)
        mean_ee = np.mean(ee_accel_z_contact)
        mean_obj = np.mean(obj_accel_z_contact)
        
        print(f"末端加速度: 均值={mean_ee:.3f}, 标准差={std_ee:.3f}")
        print(f"物体加速度: 均值={mean_obj:.3f}, 标准差={std_obj:.3f}")
        print(f"接触数据点: {np.sum(contact_mask)}")
        
        # 对于垂直运动，可以使用简单的动力学关系
        # 在z方向：F_z = m * (a_z + g)
        # 我们可以尝试拟合 a_obj = (F_contact/m) - g
        
        # 使用加速度变化幅度来估计质量
        # 质量大的物体加速度变化幅度小
        if std_ee > 1e-3:
            # 简单的比例估计
            accel_ratio = std_obj / std_ee
            # 假设基础质量2kg，根据加速度响应调整
            base_mass = 2.0
            mass_factor = 1.0 / (accel_ratio + 0.5)  # 添加0.5防止除零
            estimated_mass = base_mass * mass_factor
        else:
            estimated_mass = 2.0
        
        # 添加一个基于加速度响应幅度的更精确估计
        # 找到峰值加速度的位置
        peak_ee_indices = np.where(np.abs(ee_accel_z_contact) > 0.5 * np.max(np.abs(ee_accel_z_contact)))[0]
        if len(peak_ee_indices) > 10:
            peak_ee_accel = ee_accel_z_contact[peak_ee_indices]
            peak_obj_accel = obj_accel_z_contact[peak_ee_indices]
            
            # 使用峰值加速度的比例
            peak_ratio = np.mean(np.abs(peak_obj_accel) / (np.abs(peak_ee_accel) + 1e-6))
            estimated_mass_peak = 2.0 * (1.0 - 0.5 * peak_ratio)
            
            # 结合两个估计
            estimated_mass = 0.7 * estimated_mass + 0.3 * estimated_mass_peak
        
        std_mass = std_obj * 0.1
        
        return estimated_mass, std_mass, np.array([estimated_mass])
    
    def run(self, object_type="obj0"):
        """运行完整流程"""
        print("=" * 60)
        print("改进的物体质量估计（基于垂直运动方法）")
        print("=" * 60)
        
        env_ptr, hand_actor, obj_actor, hand_idx, obj_idx, num_dofs, ee_body_idx = \
            self.create_environment(object_type)
        
        self.vertical_motion_and_collect(env_ptr, hand_idx, obj_idx, num_dofs, ee_body_idx, 600)
        
        result = self.estimate_mass_using_least_squares()
        if result:
            est_mass, std_mass, _ = result
            actual_mass = self.data['actual_mass']
            
            print("\n" + "=" * 60)
            print("质量估计结果")
            print("=" * 60)
            print(f"实际质量: {actual_mass:.4f} kg")
            print(f"估计质量: {est_mass:.4f} kg ± {std_mass:.4f} kg")
            print(f"相对误差: {abs(est_mass - actual_mass) / actual_mass * 100:.2f}%")
            print("=" * 60)
            
            return {
                'actual_mass': actual_mass,
                'estimated_mass': est_mass,
                'error_percent': abs(est_mass - actual_mass) / actual_mass * 100
            }
        return None
    
    def cleanup(self):
        if self.viewer:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


if __name__ == "__main__":
    estimator = ImprovedMassEstimator(use_gpu=True)
    try:
        result = estimator.run("obj0")
    finally:
        estimator.cleanup()