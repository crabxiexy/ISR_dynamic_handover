# Object 和 Goal Asset 加载详解

## 概述

第480行之后的代码主要负责在仿真环境中创建和配置物体（object）和目标物体（goal object）。这部分代码分为两个阶段：

1. **临时资产创建**（480-488行）：创建临时的球体资产（实际未使用）
2. **实际物体创建**（565-601行）：在循环中为每个环境创建实际的物体和目标物体

---

## 第一部分：临时资产创建（480-488行）

```python
# load manipulated object and goal assets
object_asset_options = gymapi.AssetOptions()
object_asset_options.density = 500

self.object_radius = 0.06
object_asset = self.gym.create_sphere(self.sim, 0.12, object_asset_options)

object_asset_options.disable_gravity = True
goal_asset = self.gym.create_sphere(self.sim, 0.04, object_asset_options)
```

### 代码分析

**⚠️ 注意：这部分代码创建的资产实际上没有被使用！**

1. **创建物体资产选项**：
   - `density = 500`：设置物体密度为 500 kg/m³

2. **创建临时物体资产**：
   - `self.gym.create_sphere(self.sim, 0.12, ...)`：创建一个半径为 0.12 的球体
   - 这个球体资产被存储在 `object_asset` 变量中，但**从未被使用**

3. **创建临时目标资产**：
   - `object_asset_options.disable_gravity = True`：禁用重力（目标物体不需要受重力影响）
   - `self.gym.create_sphere(self.sim, 0.04, ...)`：创建一个半径为 0.04 的球体作为目标
   - 这个目标资产被存储在 `goal_asset` 变量中，但**从未被使用**

### 为什么存在这段代码？

这段代码可能是：
- **遗留代码**：早期版本可能使用简单的球体，后来改为使用 `object_asset_dict` 中的复杂物体
- **占位符**：为未来可能的简单物体模式预留接口

---

## 第二部分：实际物体创建（565-601行）

这部分代码在 `for i in range(self.num_envs)` 循环中执行，为每个环境创建实际的物体。

### 2.1 选择物体类型（566-567行）

```python
# add object
index = i % len(self.used_training_objects)
select_obj = self.used_training_objects[index]
```

**功能**：
- 使用模运算循环选择物体类型
- 例如：如果有 8 种训练物体，环境 0-7 使用物体 0-7，环境 8-15 再次使用物体 0-7
- 这样可以确保所有环境均匀分布不同的物体类型

**示例**：
```python
self.used_training_objects = ["obj0", "obj1", "obj2", "obj4", "obj6", "obj7", "obj9", "obj10"]
# 环境 0 -> obj0
# 环境 1 -> obj1
# ...
# 环境 7 -> obj10
# 环境 8 -> obj0 (循环)
```

---

### 2.2 创建实际物体（568-575行）

```python
object_handle = self.gym.create_actor(env_ptr, self.object_asset_dict[select_obj]['obj'], 
                                      object_start_pose, "object", i, 0, 0)

self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                               object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z, object_start_pose.r.w,
                               0, 0, 0, 0, 0, 0])
object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
self.object_indices.append(object_idx)
```

**功能**：
1. **创建物体 Actor**：
   - 使用 `object_asset_dict[select_obj]['obj']` 中预加载的物体资产
   - 位置：`object_start_pose`（在第506行定义：`gymapi.Vec3(0.025, -0.38, 0.449)`）
   - 名称：`"object"`，ID：`i`

2. **保存初始状态**：
   - 记录物体的初始位置（x, y, z）和旋转（四元数：x, y, z, w）
   - 初始速度全部为 0（最后6个0：线速度 xyz，角速度 xyz）

3. **获取索引**：
   - 获取物体在仿真中的全局索引，用于后续访问物体状态

---

### 2.3 配置物体物理属性（577-591行）

```python
# 配置刚体属性（质量）
lego_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
for lego_body_prop in lego_body_props:
    lego_body_prop.mass *= 1  # 乘以1，实际上没有改变
self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, lego_body_props)

# 配置形状属性（弹性）
object_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
for object_shape_prop in object_shape_props:
    object_shape_prop.restitution = 0  # 设置为完全非弹性（无反弹）
self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_shape_props)

# 配置手部形状属性（应用到物体上？这里可能有bug）
hand_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, allegro_hand_actor)
for hand_shape_prop in hand_shape_props:
    hand_shape_prop.restitution = 0.
self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, hand_shape_props)  # ⚠️ 注意：这里应用到object_handle
```

**功能**：
1. **质量配置**：
   - `mass *= 1`：实际上没有改变质量（可能是占位符代码）

2. **弹性配置**：
   - `restitution = 0`：完全非弹性碰撞（物体碰撞时不反弹）

3. **⚠️ 潜在问题**：
   - 第588-591行获取手部的形状属性，但应用到物体上
   - 这可能是代码错误，或者是有意为之（让物体使用手部的弹性属性）

---

### 2.4 创建目标物体（Goal Object）（593-596行）

```python
# add goal object
goal_handle = self.gym.create_actor(env_ptr, self.object_asset_dict[select_obj]['goal'], 
                                     goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
self.goal_object_indices.append(goal_object_idx)
```

**功能**：
1. **创建目标物体**：
   - 使用 `object_asset_dict[select_obj]['goal']` 中的目标资产
   - 位置：`goal_start_pose`（在第514-517行定义）
   - 名称：`"goal_object"`，ID：`i + self.num_envs`（确保ID唯一）

2. **目标物体的特点**：
   - 在 `create_object_asset_dict` 中创建时设置了 `disable_gravity = True`
   - 因此目标物体不受重力影响，保持固定位置
   - 用于可视化任务目标位置

---

### 2.5 创建预测目标物体（Predict Goal Object）（598-602行）

```python
# add goal object
predict_goal_handle = self.gym.create_actor(env_ptr, self.object_asset_dict[select_obj]['predict goal'], 
                                            goal_start_pose, "predict_goal_object", i + self.num_envs * 2, 0, 0)
predict_goal_object_idx = self.gym.get_actor_index(env_ptr, predict_goal_handle, gymapi.DOMAIN_SIM)
self.predict_goal_object_indices.append(predict_goal_object_idx)
self.gym.set_rigid_body_color(env_ptr, predict_goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.4, 0.))
```

**功能**：
1. **创建预测目标物体**：
   - 使用 `object_asset_dict[select_obj]['predict goal']` 中的预测目标资产
   - 位置：`goal_start_pose`（初始位置，后续会动态更新）
   - 名称：`"predict_goal_object"`，ID：`i + self.num_envs * 2`（确保ID唯一）

2. **可视化设置**：
   - `set_rigid_body_color(..., gymapi.Vec3(0.8, 0.4, 0.))`：设置为橙色（RGB: 0.8, 0.4, 0）
   - 用于可视化轨迹估计器预测的交接位置

3. **用途**：
   - 在 `pre_physics_step` 中会动态更新这个物体的位置（第1056行）
   - 显示神经网络预测的物体交接位置

---

## 三种物体的对比

| 物体类型 | 资产来源 | 用途 | 是否受重力 | 是否移动 | 颜色 |
|---------|---------|------|-----------|---------|------|
| **Object** | `object_asset_dict[obj]['obj']` | 实际操作的物体 | ✅ 是 | ✅ 是 | 默认 |
| **Goal Object** | `object_asset_dict[obj]['goal']` | 显示任务目标位置 | ❌ 否 | ❌ 否 | 默认 |
| **Predict Goal** | `object_asset_dict[obj]['predict goal']` | 显示预测交接位置 | ❌ 否 | ✅ 是（动态更新） | 橙色 |

---

## 资产字典结构

在 `create_object_asset_dict` 方法中（364-380行），为每个训练物体创建了三种资产：

```python
self.object_asset_dict[used_objects] = {
    'obj': self.object_asset,           # 实际物体（受重力）
    'goal': goal_asset,                 # 目标物体（不受重力）
    'predict goal': predict_goal_asset  # 预测目标（不受重力）
}
```

**为什么需要三种资产？**
- **obj**：需要受重力影响，参与物理仿真
- **goal**：不需要受重力，作为静态目标标记
- **predict goal**：不需要受重力，动态显示预测位置

---

## 关键位置定义

### Object 初始位置（506行）
```python
object_start_pose.p = gymapi.Vec3(0.025, -0.38, 0.449)
```
- x: 0.025（稍微偏右）
- y: -0.38（在左手前方）
- z: 0.449（高度）

### Goal 初始位置（514-517行）
```python
goal_start_pose.p = object_start_pose.p + self.goal_displacement
goal_start_pose.p.z -= 0.0  # 实际上没有改变
```
- 初始位置与物体相同（`goal_displacement = Vec3(0, 0, 0)`）
- 后续在 `reset_target_pose` 中会随机化目标位置

---

## 总结

第480行之后的代码主要完成以下任务：

1. **创建临时资产**（未使用）：创建简单的球体资产作为占位符
2. **循环创建物体**：为每个环境创建三种物体
   - **Object**：实际操作的物体（受重力，可移动）
   - **Goal Object**：任务目标标记（不受重力，静态）
   - **Predict Goal**：预测位置可视化（不受重力，动态更新）
3. **配置物理属性**：设置质量、弹性等物理参数
4. **保存索引**：记录每个物体的索引，用于后续访问

**关键点**：
- 每个环境使用不同的物体类型（通过模运算循环分配）
- 三种物体使用相同的几何形状，但不同的物理属性
- 目标物体和预测目标物体不受重力，只用于可视化

