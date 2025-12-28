# logdir路径问题修复说明

## 问题描述

无论设置什么`logdir`参数，打印的`logdir`是正确的，但实际模型保存的路径总是固定的：
`/home/caozhuo/dynamic_handover/dexteroushandenvs/logs/allegro_hand_dynamic_handover/mappo`

## 问题根源

### 调用链分析

1. **train.py (第72行)**
   ```python
   cfg, cfg_train, logdir = load_cfg(args)
   print("logdir:", logdir)  # 这里打印的是正确的logdir
   ```

2. **train.py (第37行)**
   ```python
   runner = process_MultiAgentRL(args, env=env, config=cfg_train, model_dir=args.model_dir)
   ```
   这里传入的是`cfg_train`，而不是`logdir`

3. **runner.py (第66行)**
   ```python
   self.run_dir = config["run_dir"]  # 这里从cfg_train中读取run_dir
   ```

4. **cfg/mappo/config.yaml (第4行)**
   ```yaml
   run_dir: ./logs  # 这是硬编码的值！
   ```

### 问题所在

- `logdir`在`load_cfg`中正确计算并返回
- 但`cfg_train["run_dir"]`仍然是从YAML文件读取的硬编码值`./logs`
- `runner.py`使用的是`config["run_dir"]`（即`cfg_train["run_dir"]`），而不是`logdir`

## 解决方案

在`utils/config.py`的`load_cfg`函数中，将计算得到的`logdir`覆盖到`cfg_train["run_dir"]`中：

```python
def load_cfg(args, use_rlg_config=False):
    # ... 前面的代码 ...
    
    logdir = os.path.realpath(log_id)
    
    # 重要：将logdir设置到cfg_train["run_dir"]中，以便runner使用
    # runner.py中的self.run_dir = config["run_dir"]会使用这个值
    if "run_dir" in cfg_train:
        cfg_train["run_dir"] = logdir
    else:
        cfg_train["run_dir"] = logdir
    
    return cfg, cfg_train, logdir
```

## 修复后的行为

1. 命令行参数`--logdir`被正确解析
2. `load_cfg`计算正确的`logdir`路径
3. `logdir`被设置到`cfg_train["run_dir"]`中
4. `runner.py`从`config["run_dir"]`读取正确的路径
5. 模型保存到正确的目录

## 验证

修复后，运行：
```bash
python train.py --task AllegroHandDynamicHandover --algo mappo --logdir /your/custom/path
```

模型应该保存到：
```
/your/custom/path/allegro_hand_dynamic_handover/mappo/models_seed{seed}/
```

而不是：
```
/home/caozhuo/dynamic_handover/dexteroushandenvs/logs/allegro_hand_dynamic_handover/mappo/models_seed{seed}/
```

