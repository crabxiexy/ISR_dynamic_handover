# TrajEstimator 保存逻辑说明

## 当前保存逻辑

### 保存位置
**文件**: `allegro_hand_dynamic_handover.py` 第 1075-1078 行

```python
if self.total_steps % (200 * (self.max_episode_length - 1)) == 0:
    iter = int(self.total_steps / (200 * (self.max_episode_length - 1)))
    if not self.is_test:
        torch.save(self.traj_estimator.state_dict(), self.traj_estimator_save_path + "/model.pt")
```

### 保存条件

**保存频率**: 每 `200 * (max_episode_length - 1)` 步保存一次

**计算示例**:
- 如果 `max_episode_length = 60`（默认值）
- 保存间隔 = `200 * (60 - 1) = 200 * 59 = 11,800` 步

### 保存时机

1. **触发位置**: 在 `pre_physics_step` 方法中每次调用时检查
2. **保存条件**: 
   - `self.total_steps % (200 * (self.max_episode_length - 1)) == 0`
   - `self.is_test == False`（不在测试模式下）

### 步数计算

- `self.total_steps`: 累计的总仿真步数（每次 `post_physics_step` 中递增）
- `max_episode_length`: 每个episode的最大步数（从配置文件中读取）

### 示例

假设 `max_episode_length = 60`:
- 第 11,800 步时保存（第1次）
- 第 23,600 步时保存（第2次）
- 第 35,400 步时保存（第3次）
- ...

### 转换为Episode数

如果想知道相当于多少个episode：
- 每个episode = 60步
- 11,800步 = 11,800 / 60 ≈ 196.7 个episode

所以大约每 **197个episode** 保存一次（当 `max_episode_length = 60` 时）

### 保存路径

- **路径**: `{logdir}/traj_e/model.pt`
- **注意**: 每次保存都会**覆盖**之前的模型文件（不保存历史版本）

## 对比：Action Model 保存逻辑

Action Model 的保存逻辑：
- **频率**: 每 `save_interval` 个episode保存一次（例如每500个episode）
- **保存位置**: `runner.py` 的 `save()` 方法
- **路径**: `{logdir}/allegro_hand_dynamic_handover/mappo/models_seed{seed}/{episode}/...`
- **特点**: 按episode编号保存，可以保留多个版本

## 主要区别

| 特性 | TrajEstimator | Action Model |
|------|--------------|--------------|
| **保存单位** | 步数（steps） | Episode数 |
| **保存频率** | 每 11,800 步（约197个episode） | 每 500 个episode |
| **版本管理** | ❌ 覆盖保存 | ✅ 按episode保存多个版本 |
| **保存位置** | `pre_physics_step` | `runner.save()` |
| **可配置性** | ❌ 硬编码 | ✅ 通过 `save_interval` 配置 |

## 建议

如果需要让 TrajEstimator 与 Action Model 同步保存（相同频率），需要：
1. 修改保存逻辑，改为基于episode而不是步数
2. 从 runner 中调用保存方法，使用相同的episode编号
3. 按episode创建子文件夹，避免覆盖历史版本


