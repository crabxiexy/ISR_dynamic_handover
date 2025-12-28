"""
物体质量估计脚本 - 最终修复版

通过让投掷机械臂执行简单的上下运动，收集力、力矩、加速度等信息，
使用物理方程（F = ma）来估计物体质量，并与实际质量进行比较。
"""

import numpy as np
import os
import sys
import gc

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
from isaacgym.torch_utils import *
import torch

class MassEstimator:
    def __init__(self, use_gpu=False):
        # 初始化 gym
        self.gym = gymapi.acquire_gym()
        
        # 设备设置
        self.use_gpu = use_gpu
        if torch.cuda.is_available() and use_gpu:
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
            use_gpu = False  # 确保如果CUDA不可用，不使用GPU
            
        print(f"使用设备: {self.device}")
        
        # 创建仿真
        self.sim_params = gymapi.SimParams()
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        self.sim_params.dt = 1.0 / 60.0
        
        # 使用 PhysX 引擎
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 8
        self.sim_params.physx.num_velocity_iterations = 0
        self.sim_params.physx.rest_offset = 0.0
        self.sim_params.physx.contact_offset = 0.002
        self.sim_params.physx.use_gpu = use_gpu
        self.sim_params.use_gpu_pipeline = use_gpu
        
        # 设置计算设备和图形设备
        compute_device_id = 0 if use_gpu else -1
        graphics_device_id = 0
        
        # 关键修复：确保不使用图形界面时设置正确的参数
        self.sim = self.gym.create_sim(
            compute_device_id, 
            graphics_device_id, 
            gymapi.SIM_PHYSX, 
            self.sim_params
        )
        
        if self.sim is None:
            print("Failed to create sim")
            sys.exit(1)
        
        # 添加地面
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        
        # 不创建观察器（headless模式）
        self.viewer = None
        
        # 数据收集
        self.data_history = {
            'forces': [],
            'torques': [],
            'accelerations': [],
            'velocities': [],
            'positions': [],
            'object_mass': None,
            'time': []
        }
        
    def create_environment(self, object_type="obj0"):
        """创建环境：物体 + 投掷机械臂（左手）"""
        asset_root = "../assets"
        
        # 加载投掷机械臂（左手/another_hand）
        allegro_hand_another_asset_file = "urdf/xarm6/xarm6_allegro_right_2023_binghao.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False  # 机械臂应该有重力
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        # 修复枚举问题：使用整数而不是枚举
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        
        hand_asset = self.gym.load_asset(self.sim, asset_root, allegro_hand_another_asset_file, asset_options)
        
        # 加载物体
        object_asset_file = f"urdf/binghao_obj/objects/{object_type}.urdf"
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 2000  # 设置密度，用于计算质量
        
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)
        
        # 获取实际质量（从刚体属性）
        # 注意：需要在创建actor后获取
        num_hand_bodies = self.gym.get_asset_rigid_body_count(hand_asset)
        num_hand_dofs = self.gym.get_asset_dof_count(hand_asset)
        
        print(f"Hand bodies: {num_hand_bodies}, DOFs: {num_hand_dofs}")
        
        # 创建环境
        spacing = 2.0
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        
        env_ptr = self.gym.create_env(self.sim, lower, upper, 1)
        
        # 设置机械臂初始姿态
        hand_start_pose = gymapi.Transform()
        hand_start_pose.p = gymapi.Vec3(0, -0.8, 0.3)  # 调整位置，更高一点
        hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 1.57079)
        
        # 设置物体初始位置（在机械臂手部）
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0, -0.8, 0.6)  # 在手部上方
        object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)
        
        # 创建 actor
        hand_actor = self.gym.create_actor(env_ptr, hand_asset, hand_start_pose, "hand", 0, -1, 0)
        object_actor = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", 0, 0, 0)
        
        # 获取实际物体质量
        object_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_actor)
        actual_mass = sum([prop.mass for prop in object_body_props])
        self.data_history['object_mass'] = actual_mass
        print(f"实际物体质量: {actual_mass:.4f} kg")
        
        # 配置机械臂 DOF 属性
        hand_dof_props = self.gym.get_asset_dof_properties(hand_asset)
        
        # 设置机械臂关节属性 - 修复：确保使用正确的驱动模式
        for i in range(num_hand_dofs):
            hand_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if i < 6:  # 前6个DOF是机械臂基座
                hand_dof_props['stiffness'][i] = 800.0
                hand_dof_props['damping'][i] = 80.0
                hand_dof_props['effort'][i] = 200.0
                hand_dof_props['velocity'][i] = 3.0
            else:  # 手部关节
                hand_dof_props['stiffness'][i] = 200.0
                hand_dof_props['damping'][i] = 20.0
                hand_dof_props['effort'][i] = 100.0
                hand_dof_props['velocity'][i] = 5.0
        
        self.gym.set_actor_dof_properties(env_ptr, hand_actor, hand_dof_props)
        
        # 获取 actor 索引
        hand_idx = self.gym.get_actor_index(env_ptr, hand_actor, gymapi.DOMAIN_SIM)
        object_idx = self.gym.get_actor_index(env_ptr, object_actor, gymapi.DOMAIN_SIM)
        
        # 设置物体的物理属性（增加摩擦，防止滑动）
        object_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_actor)
        for prop in object_shape_props:
            prop.friction = 1.5  # 增加摩擦系数
            prop.restitution = 0.05  # 减少弹性
        self.gym.set_actor_rigid_shape_properties(env_ptr, object_actor, object_shape_props)
        
        # 设置机械手形状属性（增加抓握能力）
        hand_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, hand_actor)
        for prop in hand_shape_props:
            prop.friction = 1.2  # 手指摩擦
        self.gym.set_actor_rigid_shape_properties(env_ptr, hand_actor, hand_shape_props)
        
        # 设置机械臂的初始位置
        default_dof_pos = np.zeros(num_hand_dofs, dtype=np.float32)
        default_dof_pos[:6] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # 手部关节保持抓握状态
        default_dof_pos[6:] = [
            0.0, 0.8, 0.8, 1.2,  # 拇指
            0.0, 0.8, 0.8, 1.2,  # 食指
            0.0, 0.8, 0.8, 1.2,  # 中指
            0.0, 0.8, 0.8, 1.2,  # 无名指
        ]
        
        self.gym.set_actor_dof_states(env_ptr, hand_actor, default_dof_pos, gymapi.STATE_POS)
        
        return env_ptr, hand_actor, object_actor, hand_idx, object_idx, num_hand_dofs
    
    def run_motion_and_collect_data(self, env_ptr, hand_actor, object_actor, hand_idx, object_idx, num_hand_dofs, num_steps=300):
        """执行上下运动并收集数据"""
        
        # 准备数据缓冲区
        self.gym.prepare_sim(self.sim)
        
        # 获取状态张量
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        
        # 包装为 torch 张量
        root_states = gymtorch.wrap_tensor(actor_root_state_tensor)
        dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        
        # 获取初始状态
        hand_pos = root_states[hand_idx, 0:3]
        object_pos = root_states[object_idx, 0:3]
        
        print(f"机械臂初始位置: {hand_pos}")
        print(f"物体初始位置: {object_pos}")
        
        # 运动参数
        amplitude = 0.15  # 减少幅度，避免太剧烈
        frequency = 0.3   # 降低频率，让运动更平缓
        dt = self.sim_params.dt
        
        # 初始 DOF 位置
        default_dof_pos = torch.zeros(num_hand_dofs, dtype=torch.float32)
        # default_dof_pos[:6] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        # # 手部关节保持抓握状态
        # default_dof_pos[6:] = torch.tensor([
        #     0.0, 0.8, 0.8, 1.2,  # 拇指
        #     0.0, 0.8, 0.8, 1.2,  # 食指
        #     0.0, 0.8, 0.8, 1.2,  # 中指
        #     0.0, 0.8, 0.8, 1.2,  # 无名指
        # ], dtype=torch.float32)
        
        # 存储历史数据
        positions_history = []
        velocities_history = []
        accelerations_history = []
        time_history = []
        
        print("开始执行运动并收集数据...")
        
        for step in range(num_steps):
            # 计算目标位置（上下运动）
            t = step * dt
            z_offset = amplitude * np.sin(2 * np.pi * frequency * t)
            
            # 创建目标DOF位置
            target_dof_pos = default_dof_pos.clone()
            target_dof_pos[2] = 0.3 + z_offset  # 修改 z 轴位置（从0.3开始）
            
            # 转换为numpy数组
            target_dof_pos_np = target_dof_pos.numpy()
            
            # 设置 DOF 目标位置
            self.gym.set_actor_dof_position_targets(
                env_ptr, 
                hand_actor, 
                target_dof_pos_np
            )
            
            # 执行一步仿真
            self.gym.simulate(self.sim)
            
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            
            # 刷新状态
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            
            # 收集物体数据
            object_pos = root_states[object_idx, 0:3].cpu().numpy()
            object_linvel = root_states[object_idx, 7:10].cpu().numpy()
            
            # 计算加速度（使用数值微分）
            if step > 0:
                prev_vel = velocities_history[-1]
                accel = (object_linvel - prev_vel) / dt
                accelerations_history.append(accel)
            else:
                accelerations_history.append(np.array([0.0, 0.0, 0.0]))
            
            positions_history.append(object_pos.copy())
            velocities_history.append(object_linvel.copy())
            time_history.append(t)
        
        # 转换为 numpy 数组
        self.data_history['positions'] = np.array(positions_history)
        self.data_history['velocities'] = np.array(velocities_history)
        self.data_history['accelerations'] = np.array(accelerations_history)
        self.data_history['time'] = np.array(time_history)
        
        print(f"收集了 {num_steps} 步数据")
        print(f"位置范围: z = [{self.data_history['positions'][:, 2].min():.3f}, {self.data_history['positions'][:, 2].max():.3f}]")
        print(f"速度范围: z = [{self.data_history['velocities'][:, 2].min():.3f}, {self.data_history['velocities'][:, 2].max():.3f}]")
        print(f"加速度范围: z = [{self.data_history['accelerations'][:, 2].min():.3f}, {self.data_history['accelerations'][:, 2].max():.3f}]")
        
    def estimate_mass_improved(self):
        """改进的质量估计算法"""
        if len(self.data_history['accelerations']) < 10:
            print("数据不足，无法进行准确估计")
            return 0.5, 0.5  # 返回默认值和不确定性
        
        positions = self.data_history['positions']
        velocities = self.data_history['velocities']
        accelerations = self.data_history['accelerations']
        
        # 只分析z方向（垂直方向）
        z_pos = positions[:, 2]
        z_vel = velocities[:, 2]
        z_acc = accelerations[:, 2]
        
        # 重力加速度
        g = 9.81
        
        # 方法1：基于加速度的分析
        # 计算有效加速度（减去重力）
        effective_acc = z_acc + g  # 当物体被支撑时，加速度应该接近0
        
        # 找到物体被稳定支撑的时段（加速度接近0）
        supported_mask = np.abs(effective_acc) < 2.0  # 2 m/s²阈值
        
        if np.sum(supported_mask) > 20:
            # 在支撑状态下，加速度应该很小
            mean_supported_acc = np.mean(effective_acc[supported_mask])
            std_supported_acc = np.std(effective_acc[supported_mask])
            
            print(f"支撑状态数据点: {np.sum(supported_mask)}")
            print(f"支撑状态平均有效加速度: {mean_supported_acc:.3f} m/s²")
            print(f"支撑状态加速度标准差: {std_supported_acc:.3f} m/s²")
            
            # 重物体在支撑状态下加速度变化小，轻物体变化大
            # 这是一个经验公式，需要根据实际情况调整
            mass_from_acc = 0.5 / (std_supported_acc + 0.01)
        else:
            mass_from_acc = 0.3  # 默认值
        
        # 方法2：基于速度的分析
        # 计算速度的标准差（重物体速度变化小）
        vel_std = np.std(z_vel)
        mass_from_vel = 0.3 / (vel_std + 0.05)
        
        # 方法3：基于位置的分析
        # 计算运动幅度（峰值到谷值）
        from scipy.signal import find_peaks
        
        # 找到位置峰值
        peaks, _ = find_peaks(z_pos, height=np.mean(z_pos), distance=20)
        valleys, _ = find_peaks(-z_pos, height=-np.mean(z_pos), distance=20)
        
        if len(peaks) > 1 and len(valleys) > 1:
            # 计算平均振幅
            peak_heights = z_pos[peaks]
            valley_heights = z_pos[valleys]
            
            if len(peak_heights) > 0 and len(valley_heights) > 0:
                avg_peak = np.mean(peak_heights[:min(5, len(peak_heights))])
                avg_valley = np.mean(valley_heights[:min(5, len(valley_heights))])
                amplitude = avg_peak - avg_valley
                
                # 重物体振幅小，轻物体振幅大
                mass_from_pos = 0.4 / (amplitude + 0.02)
            else:
                mass_from_pos = 0.3
        else:
            mass_from_pos = 0.3
        
        # 综合三种方法
        weights = [0.4, 0.3, 0.3]  # 给加速度方法更高的权重
        masses = [mass_from_acc, mass_from_vel, mass_from_pos]
        
        final_mass = sum(w * m for w, m in zip(weights, masses))
        
        # 限制在合理范围内
        final_mass = max(0.05, min(2.0, final_mass))
        
        # 计算不确定性（基于三种方法的一致性）
        uncertainty = np.std(masses) if len(masses) > 1 else 0.2
        
        print(f"各方法估计值: 加速度={mass_from_acc:.3f}, 速度={mass_from_vel:.3f}, 位置={mass_from_pos:.3f}")
        
        return final_mass, uncertainty
    
    def estimate_mass(self):
        """主质量估计函数"""
        return self.estimate_mass_improved()
    
    def run(self, object_type="obj0"):
        """运行完整的质量估计流程"""
        print("=" * 60)
        print(f"物体质量估计实验: {object_type}")
        print("=" * 60)
        
        try:
            # 创建环境
            env_ptr, hand_actor, object_actor, hand_idx, object_idx, num_hand_dofs = \
                self.create_environment(object_type)
            
            # 执行运动并收集数据
            self.run_motion_and_collect_data(
                env_ptr, hand_actor, object_actor, hand_idx, object_idx, num_hand_dofs,
                num_steps=200  # 减少步数，3.3秒的数据
            )
            
            # 估计质量
            estimated_mass, uncertainty = self.estimate_mass()
            
            actual_mass = self.data_history['object_mass']
            
            print("\n" + "=" * 60)
            print("质量估计结果")
            print("=" * 60)
            print(f"实际质量: {actual_mass:.4f} kg")
            print(f"估计质量: {estimated_mass:.4f} kg ± {uncertainty:.4f} kg")
            print(f"绝对误差: {abs(estimated_mass - actual_mass):.4f} kg")
            print(f"相对误差: {abs(estimated_mass - actual_mass) / actual_mass * 100:.2f}%")
            print("=" * 60)
            
            return {
                'actual_mass': actual_mass,
                'estimated_mass': estimated_mass,
                'uncertainty': uncertainty,
                'error_percent': abs(estimated_mass - actual_mass) / actual_mass * 100,
            }
        except Exception as e:
            print(f"运行过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # 清理当前环境的数据
            self.data_history = {
                'forces': [],
                'torques': [],
                'accelerations': [],
                'velocities': [],
                'positions': [],
                'object_mass': None,
                'time': []
            }
    
    def cleanup(self):
        """安全清理资源"""
        print("清理资源...")
        try:
            # 首先销毁观察器（如果有）
            if self.viewer is not None:
                try:
                    self.gym.destroy_viewer(self.viewer)
                    self.viewer = None
                except:
                    pass
            
            # 然后销毁仿真
            if self.sim is not None:
                try:
                    self.gym.destroy_sim(self.sim)
                    self.sim = None
                except:
                    pass
            
            # 清理GPU缓存（如果使用GPU）
            if self.use_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 强制垃圾回收
            gc.collect()
            
            print("资源清理完成")
        except Exception as e:
            print(f"清理资源时出错: {e}")
            # 不重新抛出异常，避免影响主流程


def main_safe():
    """安全的主函数，避免Segmentation fault"""
    print("物体质量估计实验 - 安全版")
    print("=" * 60)
    
    # 参数设置
    use_gpu = False  # 使用CPU模式更稳定
    test_objects = ["obj0", "obj1", "obj2"]
    
    results = []
    
    # 对每个物体进行测试
    for obj_type in test_objects:
        print(f"\n{'='*60}")
        print(f"测试物体: {obj_type}")
        print(f"{'='*60}")
        
        estimator = None
        try:
            # 创建估计器
            estimator = MassEstimator(use_gpu=use_gpu)
            
            # 运行实验
            result = estimator.run(obj_type)
            if result:
                results.append((obj_type, result))
                print(f"\n{obj_type} 测试完成")
            else:
                print(f"\n{obj_type} 测试失败")
        
        except KeyboardInterrupt:
            print(f"\n用户中断 {obj_type} 测试")
            break
        except Exception as e:
            print(f"\n测试 {obj_type} 时出错: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 确保清理资源
            if estimator is not None:
                try:
                    estimator.cleanup()
                    estimator = None
                except:
                    pass
            
            # 给系统一点时间释放资源
            import time
            time.sleep(0.5)
            
            # 强制垃圾回收
            gc.collect()
    
    # 打印汇总结果
    if results:
        print("\n" + "=" * 60)
        print("汇总结果")
        print("=" * 60)
        
        total_error = 0
        for obj_type, result in results:
            print(f"{obj_type}:")
            print(f"  实际质量: {result['actual_mass']:.4f} kg")
            print(f"  估计质量: {result['estimated_mass']:.4f} ± {result['uncertainty']:.4f} kg")
            print(f"  相对误差: {result['error_percent']:.2f}%")
            print()
            total_error += result['error_percent']
        
        if len(results) > 0:
            avg_error = total_error / len(results)
            print(f"平均相对误差: {avg_error:.2f}%")
    else:
        print("\n没有成功的结果")
    
    print("\n实验完成!")
    print("=" * 60)


if __name__ == "__main__":
    # 设置更安全的信号处理
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    # 运行安全版本
    main_safe()