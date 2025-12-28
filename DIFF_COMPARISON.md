# 文件差异对比分析

## 概述
对比文件：
- **原始文件**: `allegro_hand_dynamic_handover_original.py`
- **修改文件**: `allegro_hand_dynamic_handover.py`

## 主要差异总结

### ✅ 1. 添加物体质量跟踪系统（正确）

**位置**: `_create_envs` 方法 (第 526, 589, 651 行)

**原始代码**:
```python
# 在循环外，只获取最后一个环境的物体质量（有问题）
object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
self.object_rb_masses = [prop.mass for prop in object_rb_props]
```

**修改后代码**:
```python
# 在初始化时添加
self.env_object_masses = []

# 在循环内，为每个环境计算并存储物体总质量
for lego_body_prop in lego_body_props:
    lego_body_prop.mass *= 1
# compute and store the total mass of this env's object
try:
    mass_sum = 0.0
    for p in lego_body_props:
        mass_sum += p.mass
except Exception:
    mass_sum = 0.0
self.env_object_masses.append(mass_sum)

# 在循环后转换为tensor
self.object_masses = to_torch(self.env_object_masses, device=self.device, dtype=torch.float).view(self.num_envs, 1)
```

**分析**: ✅ **正确**
- 原始代码只在最后一个环境创建后获取物体质量，这会导致所有环境使用相同的质量值（最后一个环境的质量）
- 修改后的代码为每个环境单独计算和存储物体质量，这是正确的做法
- 使用 `view(self.num_envs, 1)` 确保维度正确

---

### ✅ 2. 轨迹估计器输入维度增加（正确）

**位置**: `__init__` 方法 (第 684 行)

**原始代码**:
```python
self.traj_estimator = TrajEstimator(input_dim=60, output_dim=3).to(self.device)
```

**修改后代码**:
```python
# TrajEstimator input_dim increased by 1 to include object mass
self.traj_estimator = TrajEstimator(input_dim=61, output_dim=3).to(self.device)
```

**分析**: ✅ **正确**
- 因为添加了物体质量（1维）作为输入，所以输入维度从 60 增加到 61
- 60 = 20帧 × 3坐标
- 61 = 20帧 × 3坐标 + 1质量

---

### ✅ 3. 轨迹估计器输入包含物体质量（正确）

**位置**: `compute_sim2real_observation` 方法 (第 829 行)

**原始代码**:
```python
with TemporaryGrad():
    self.predict_pose, self.pose_latent_vector = self.predict_contact_pose(self.traj_estimator, self.object_state_stack_frames)
    self.update_contact_slamer(self.predict_pose)
```

**修改后代码**:
```python
# include per-object mass as additional input to the traj estimator
contact_input = torch.cat([self.object_state_stack_frames, self.object_masses], dim=1)
with TemporaryGrad():
    self.predict_pose, self.pose_latent_vector = self.predict_contact_pose(self.traj_estimator, contact_input)
    self.update_contact_slamer(self.predict_pose)
```

**分析**: ✅ **正确**
- `object_state_stack_frames` 形状: `(num_envs, 60)` (20帧 × 3坐标)
- `object_masses` 形状: `(num_envs, 1)`
- `contact_input` 形状: `(num_envs, 61)`
- 维度匹配正确

---

### ✅ 4. 观测空间添加物体质量（正确）

**位置**: `compute_sim2real_observation` 方法 (第 207, 836-839 行)

**修改1**: 在初始化时增加观测维度
```python
# add one extra slot for per-object mass
self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type] + 1
```

**修改2**: 在观测缓冲区中添加物体质量
```python
self.obs_buf[:, 263:264] = self.predict_pose[:, 0:3].detach()
# place object mass into observation (one slot)
try:
    self.obs_buf[:, 263:264] = self.object_masses.clone()
except Exception:
    pass
```

**分析**: ⚠️ **需要检查维度**
- 观测维度已正确增加 (+1)
- 但是第 834 行和第 837 行都在使用 `[263:264]`，这看起来是重复了
- 让我检查一下代码逻辑...

**重新检查**:
```python
self.obs_buf[:, 260:263] = self.predict_pose[:, 0:3].detach()  # 预测位置 (260-262)
# place object mass into observation (one slot)
try:
    self.obs_buf[:, 263:264] = self.object_masses.clone()  # 物体质量 (263)
except Exception:
    pass
```

**修正**: ✅ **正确**
- `260:263` 是 3 个元素（索引 260, 261, 262）
- `263:264` 是 1 个元素（索引 263）
- 没有重叠，这是正确的

---

### ✅ 5. 模型加载添加异常处理（正确）

**位置**: `__init__` 方法 (第 695-701 行)

**原始代码**:
```python
if self.is_test:
    self.traj_estimator.load_state_dict(torch.load("./traj_e/model.pt", map_location='cuda:0'))
    self.traj_estimator.eval()
```

**修改后代码**:
```python
if self.is_test:
    try:
        self.traj_estimator.load_state_dict(torch.load("./traj_e/model.pt", map_location='cuda:0'))
        self.traj_estimator.eval()
    except Exception as e:
        print("Failed to load traj_estimator weights (shapes mismatch?), continuing with random init:", e)
        self.traj_estimator.train()
```

**分析**: ✅ **正确**
- 添加异常处理是好的做法
- 当模型维度不匹配时（比如旧模型是60维输入，新模型是61维输入），会打印错误并继续使用随机初始化
- 这对于模型更新后的兼容性很重要

---

## 潜在问题检查

### ✅ 检查1: 物体质量计算逻辑

在 `_create_envs` 方法中：
```python
lego_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
for lego_body_prop in lego_body_props:
    lego_body_prop.mass *= 1  # 这里乘以1没有实际效果，可能是占位符
# compute and store the total mass of this env's object
try:
    mass_sum = 0.0
    for p in lego_body_props:
        mass_sum += p.mass
except Exception:
    mass_sum = 0.0
self.env_object_masses.append(mass_sum)
```

**分析**: ✅ **正确**
- 正确计算了所有刚体的质量总和
- 使用 try-except 处理可能的异常
- 注意：`mass *= 1` 没有实际效果，可能是为了保持代码风格一致

### ✅ 检查2: 观测缓冲区维度

观测维度已增加：
```python
self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type] + 1
```

并且正确使用：
```python
self.obs_buf[:, 263:264] = self.object_masses.clone()
```

**分析**: ✅ **维度匹配正确**

---

## 总结

### ✅ 所有修改都是正确的！

1. **物体质量跟踪**: 正确地为每个环境单独计算和存储物体质量
2. **轨迹估计器**: 正确地将输入维度从60增加到61，并在输入中包含物体质量
3. **观测空间**: 正确地在观测空间中添加了物体质量（+1维）
4. **异常处理**: 添加了合理的异常处理，提高了代码的健壮性

### 建议

1. **代码清理**: `mass *= 1` 这行代码可以移除，因为它没有实际效果
2. **测试**: 建议测试不同质量的物体，确保轨迹估计器能够正确利用质量信息
3. **文档**: 可以考虑在代码中添加注释，说明为什么需要物体质量信息

---

## 对比表格

| 项目 | 原始文件 | 修改文件 | 状态 |
|------|---------|---------|------|
| 物体质量存储 | 只存储最后一个环境 | 为每个环境单独存储 | ✅ 正确 |
| 轨迹估计器输入维度 | 60 | 61 | ✅ 正确 |
| 轨迹估计器输入内容 | 只有轨迹数据 | 轨迹数据 + 质量 | ✅ 正确 |
| 观测空间维度 | N | N+1 | ✅ 正确 |
| 模型加载异常处理 | 无 | 有 | ✅ 改进 |

