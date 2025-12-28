# 模型保存路径说明

## 1. TrajEstimator 保存路径

### 定义位置
**文件**: `allegro_hand_dynamic_handover.py` 第 691 行

```python
self.traj_estimator_save_path = "./traj_e/"
os.makedirs(self.traj_estimator_save_path, exist_ok=True)
```

### 保存路径
- **目录**: `./traj_e/` （相对于运行脚本的当前工作目录）
- **文件名**: `model.pt`
- **完整路径**: `./traj_e/model.pt`

### 保存时机
**文件**: `allegro_hand_dynamic_handover.py` 第 1067-1070 行

```python
if self.total_steps % (200 * (self.max_episode_length - 1)) == 0:
    iter = int(self.total_steps / (200 * (self.max_episode_length - 1)))
    if not self.is_test:
        torch.save(self.traj_estimator.state_dict(), self.traj_estimator_save_path + "/model.pt")
```

**保存频率**: 
- 每 `200 * (max_episode_length - 1)` 步保存一次
- 例如：如果 `max_episode_length = 60`，则每 `200 * 59 = 11,800` 步保存一次

### 加载路径
**文件**: `allegro_hand_dynamic_handover.py` 第 697 行

```python
if self.is_test:
    try:
        self.traj_estimator.load_state_dict(torch.load("./traj_e/model.pt", map_location='cuda:0'))
        self.traj_estimator.eval()
```

**说明**: 
- 在测试模式（`is_test=True`）时，从 `./traj_e/model.pt` 加载模型
- 如果加载失败（例如维度不匹配），会打印错误信息并继续使用随机初始化

---

## 2. Action Model (策略模型) 保存路径

### 定义位置
**文件**: `algorithms/marl/runner.py` 第 71-73 行

```python
self.save_dir = str(self.run_dir + '/' + self.env_name + '/' + 
                    self.algorithm_name + '/models_seed{}'.format(self.seed))
if not os.path.exists(self.save_dir):
    os.makedirs(self.save_dir)
```

### 路径组成

```
save_dir = run_dir / env_name / algorithm_name / models_seed{seed}
```

**组成部分**:
- `run_dir`: 运行目录（从配置文件 `config["run_dir"]` 获取）
- `env_name`: 环境名称（从 `cfg["env"]["env_name"]` 获取，例如 `"allegro_hand_dynamic_handover"`）
- `algorithm_name`: 算法名称（从 `config["algorithm_name"]` 获取，例如 `"mappo"`, `"happo"`, `"hatrpo"`）
- `seed`: 随机种子（从 `cfg["seed"]` 获取）

### 示例路径

假设：
- `run_dir = "./logs"`
- `env_name = "allegro_hand_dynamic_handover"`
- `algorithm_name = "mappo"`
- `seed = 42`

则完整路径为：
```
./logs/allegro_hand_dynamic_handover/mappo/models_seed42/
```

### 保存的文件名

**文件**: `algorithms/marl/runner.py` 第 312-323 行

根据是否使用单一网络，保存的文件不同：

#### 情况1: 使用单一网络 (`use_single_network=True`)
```python
save_path = save_dir / {episode} / model_agent{agent_id}.pt
```

例如（双智能体）：
```
./logs/.../models_seed42/100/model_agent0.pt
./logs/.../models_seed42/100/model_agent1.pt
```

#### 情况2: 使用分离网络（Actor-Critic）(`use_single_network=False`)
```python
save_path = save_dir / {episode} / actor_agent{agent_id}.pt
save_path = save_dir / {episode} / critic_agent{agent_id}.pt
```

例如（双智能体）：
```
./logs/.../models_seed42/100/actor_agent0.pt
./logs/.../models_seed42/100/critic_agent0.pt
./logs/.../models_seed42/100/actor_agent1.pt
./logs/.../models_seed42/100/critic_agent1.pt
```

### 保存时机

**文件**: `algorithms/marl/runner.py` 第 312 行

```python
def save(self, episode):
    # 根据 save_interval 配置决定保存频率
```

**说明**: 
- 保存频率由配置中的 `save_interval` 决定
- `episode` 是当前训练的回合数

---

## 3. 路径配置来源

### 3.1 run_dir 配置

`run_dir` 通常通过以下方式配置：

1. **命令行参数**: 通过 `--logdir` 或类似参数传递
2. **配置文件**: 在训练配置文件中设置 `run_dir` 字段
3. **默认值**: 如果未指定，可能有默认路径

**相关代码**: `utils/config.py` 中的 `load_cfg` 函数处理日志目录设置

---

## 4. 对比总结

| 模型类型 | 保存路径 | 路径类型 | 是否可配置 |
|---------|---------|---------|-----------|
| **TrajEstimator** | `./traj_e/model.pt` | 硬编码（相对路径） | ❌ 不可配置 |
| **Action Model** | `{run_dir}/{env_name}/{algorithm_name}/models_seed{seed}/{episode}/model_agent{id}.pt` | 动态构建 | ✅ 可配置（通过 run_dir） |

---

## 5. 注意事项

### TrajEstimator 路径
- ⚠️ **硬编码**: 路径 `./traj_e/` 是硬编码的，无法通过配置文件修改
- ⚠️ **相对路径**: 使用相对路径，依赖于运行脚本的工作目录
- ⚠️ **覆盖保存**: 每次保存都会覆盖之前的 `model.pt` 文件（不保存历史版本）

### Action Model 路径
- ✅ **可配置**: `run_dir` 可以通过配置或命令行参数设置
- ✅ **结构化**: 使用清晰的目录结构组织不同环境、算法和种子的模型
- ✅ **版本管理**: 按 episode 保存，可以保留多个版本的模型
- ✅ **多智能体支持**: 为每个智能体单独保存模型文件

---

## 6. 建议

如果需要修改 TrajEstimator 的保存路径，可以：

1. **直接修改代码**: 在 `allegro_hand_dynamic_handover.py` 第 691 行修改路径
2. **通过配置**: 可以考虑将其添加到配置文件中，类似于 action model 的处理方式
3. **版本管理**: 考虑添加时间戳或迭代次数到文件名，避免覆盖历史模型


