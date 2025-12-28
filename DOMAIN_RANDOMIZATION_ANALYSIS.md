# Domain Randomization（域随机化）分析

## 结论

**代码中实现了域随机化（Domain Randomization），但目前配置中处于关闭状态。**

---

## 1. 配置状态

### 当前配置 (`allegro_hand_dynamic_handover.yaml`)

```yaml
task:
  randomize: False  # ⚠️ 当前设置为 False（未启用）
  randomization_params:
    # ... 详细的随机化参数配置 ...
```

**状态**: 域随机化功能已配置但**未启用**。

---

## 2. 代码实现

### ✅ 实现位置

#### 2.1 初始化阶段（读取配置）

**文件**: `allegro_hand_dynamic_handover.py` 第 63-64 行

```python
self.randomize = self.cfg["task"]["randomize"]
self.randomization_params = self.cfg["task"]["randomization_params"]
```

**说明**: 从配置文件中读取域随机化的开关和参数。

---

#### 2.2 重置阶段（应用随机化）

**文件**: `allegro_hand_dynamic_handover.py` 第 928-931 行

```python
def reset(self, env_ids, goal_env_ids):
    # randomization can happen only at reset time, since it can reset actor positions on GPU
    if self.randomize:
        self.apply_randomizations(self.randomization_params)
```

**说明**: 
- 在环境重置时，如果 `randomize=True`，会调用 `apply_randomizations` 方法
- 该方法是继承自 `BaseTask` 类的标准域随机化实现

---

#### 2.3 随机化缓冲区更新

**文件**: `allegro_hand_dynamic_handover.py` 第 1097 行

```python
def post_physics_step(self):
    self.progress_buf += 1
    self.randomize_buf += 1  # 更新随机化计数器
```

**说明**: 每个时间步都会更新 `randomize_buf`，用于跟踪何时需要重新随机化。

---

#### 2.4 基类实现

**文件**: `hand_base/base_task.py` 第 262 行

`apply_randomizations` 方法在基类中实现，支持以下随机化类型：

1. **观测噪声 (observations)**
   - 高斯噪声添加到观测中
   - 可配置噪声范围和分布

2. **动作噪声 (actions)**
   - 高斯噪声添加到动作中

3. **物理参数 (sim_params)**
   - 重力随机化

4. **演员参数 (actor_params)**
   - **手部 (hand)**:
     - 肌腱属性（阻尼、刚度）
     - DOF属性（阻尼、刚度、上下限）
     - 刚体属性（质量）
     - 形状属性（摩擦）
   - **物体 (object)**:
     - 缩放
     - 刚体属性（质量）
     - 形状属性（摩擦）

---

## 3. 配置的随机化参数

根据配置文件，如果启用（`randomize: True`），会随机化以下内容：

### 3.1 观测噪声
- **范围**: [0, 0.002]
- **操作**: 加法
- **分布**: 高斯分布
- **调度**: 线性（40,000步后达到最大值）

### 3.2 动作噪声
- **范围**: [0, 0.05]
- **操作**: 加法
- **分布**: 高斯分布
- **调度**: 线性（40,000步后达到最大值）

### 3.3 物理参数
- **重力**: 范围 [0, 0.4]
- **调度**: 线性（40,000步后达到最大值）

### 3.4 手部参数
- **肌腱阻尼**: [0.3, 3.0] 对数均匀分布
- **肌腱刚度**: [0.75, 1.5] 对数均匀分布
- **DOF阻尼**: [0.3, 3.0] 对数均匀分布
- **DOF刚度**: [0.75, 1.5] 对数均匀分布
- **DOF上下限**: [0, 0.01] 高斯分布
- **刚体质量**: [0.5, 1.5] 均匀分布
- **摩擦系数**: [0.7, 1.3] 均匀分布

### 3.5 物体参数
- **缩放**: [0.95, 1.05] 均匀分布
- **刚体质量**: [0.5, 1.5] 均匀分布
- **摩擦系数**: [0.7, 1.3] 均匀分布

---

## 4. 其他形式的随机化

除了域随机化，代码中还有其他形式的随机化：

### 4.1 观测中的随机噪声

**文件**: `allegro_hand_dynamic_handover.py` 第 841 行

```python
self.obs_buf[:, 248:260] = self.object_state_stack_frames[:, 36:48].clone() + rand_floats[:, 0:12] * 0.05
```

**说明**: 在观测的物体历史轨迹部分添加了固定的随机噪声（0.05 的缩放因子）。

### 4.2 重置时的随机化

**文件**: `allegro_hand_dynamic_handover.py` 第 928-956 行

- 物体初始位置随机化（通过 `reset_position_noise`）
- 物体初始旋转随机化（通过 `reset_rotation_noise`）
- 关节速度随机化（通过 `reset_dof_vel_noise`）
- 目标位置随机化（在 `reset_target_pose` 方法中）

---

## 5. 如何启用域随机化

如果要启用域随机化，只需修改配置文件：

```yaml
task:
  randomize: True  # 改为 True
  randomization_params:
    # ... 参数保持不变 ...
```

**注意事项**:
1. 域随机化会增加训练的计算开销
2. 随机化频率设置为 600 步（`frequency: 600`）
3. 大多数随机化使用线性调度，在训练的前 30,000-40,000 步逐渐增加随机化强度
4. 随机化只在环境重置时应用（PhysX 限制）

---

## 6. 总结

| 项目 | 状态 | 说明 |
|------|------|------|
| **代码实现** | ✅ 已实现 | 完整实现了域随机化框架 |
| **配置参数** | ✅ 已配置 | 详细的随机化参数配置 |
| **当前状态** | ⚠️ **未启用** | `randomize: False` |
| **继承方法** | ✅ 是 | 使用 BaseTask 的标准实现 |
| **其他随机化** | ✅ 有 | 观测噪声、重置随机化等 |

**结论**: 代码中**已实现**域随机化功能，但目前**未启用**。如果需要启用，只需将配置文件中的 `randomize: False` 改为 `randomize: True`。

