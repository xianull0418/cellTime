# AE 训练日志系统使用指南

## 📋 概述

训练系统现在使用完善的日志系统，将所有输出重定向到日志文件中，控制台只显示关键信息。

## 📁 日志文件结构

训练时会在 `{OUTPUT_DIR}/logs/` 目录下生成以下日志文件：

```
output/ae_large_scale/version3_larger/logs/
├── train_20231203_143025.log         # 主训练日志（所有信息）
├── debug_rank0_20231203_143025.log   # GPU 0 的详细 debug 日志
├── debug_rank1_20231203_143025.log   # GPU 1 的详细 debug 日志
├── debug_rank2_20231203_143025.log   # GPU 2 的详细 debug 日志
└── ...                                # 每个 GPU 一个 debug 日志
```

## 🔍 日志文件说明

### 1. `train_*.log` - 主训练日志
包含：
- 训练配置信息
- 每个 epoch 的训练和验证进度
- 模型 checkpoint 保存信息
- 重要的错误和警告

**适合查看**：训练整体进度和结果

### 2. `debug_rank*.log` - Debug 日志（每个 GPU 一个）
包含：
- 数据加载的详细信息（每个 shard 的读取进度）
- 每个 rank/worker 的文件分配情况
- 训练/验证的详细进度（每 100/20 个 batch）
- Epoch 边界标记
- 内存和性能相关的调试信息

**适合查看**：诊断卡死问题、数据加载问题

## 🚀 使用方法

### 1. 启动训练

```bash
./train_ae.sh
```

训练启动后，控制台会显示日志文件路径，然后将所有详细输出重定向到日志文件。

### 2. 实时监控训练进度

在**另一个终端**中运行：

```bash
# 监控主训练日志
tail -f output/ae_large_scale/version3_larger/logs/train_*.log

# 监控 GPU 0 的详细 debug 日志
tail -f output/ae_large_scale/version3_larger/logs/debug_rank0_*.log

# 监控所有 GPU 的 debug 日志
tail -f output/ae_large_scale/version3_larger/logs/debug_rank*.log
```

### 3. 查看特定内容

```bash
# 只看错误信息
grep ERROR output/ae_large_scale/version3_larger/logs/train_*.log

# 只看 epoch 信息
grep "EPOCH" output/ae_large_scale/version3_larger/logs/debug_rank0_*.log

# 查看数据加载完成情况
grep "COMPLETED" output/ae_large_scale/version3_larger/logs/debug_rank*.log

# 查看某个特定 rank 的文件分配
grep "Processing.*files" output/ae_large_scale/version3_larger/logs/debug_rank0_*.log
```

## 🐛 诊断卡死问题

### 步骤 1：确认训练是否卡死

```bash
# 查看最新的日志输出（如果停止更新，可能卡死）
tail output/ae_large_scale/version3_larger/logs/train_*.log

# 查看各个 rank 的最后输出
tail -n 20 output/ae_large_scale/version3_larger/logs/debug_rank*.log
```

### 步骤 2：定位卡死位置

查看 debug 日志中的 epoch 边界标记：

```bash
grep "EPOCH.*START\|EPOCH.*END" output/ae_large_scale/version3_larger/logs/debug_rank0_*.log
```

可能的卡死位置：
- **卡在 `TRAINING END` 和 `VALIDATION START` 之间** → DDP 同步问题
- **卡在 `VALIDATION END` 之后** → Checkpoint 保存或学习率调度器问题
- **某个 rank 长时间没有输出** → 该 rank 的数据加载或计算卡住

### 步骤 3：检查数据加载

```bash
# 查看各个 rank/worker 的数据分配是否均匀
grep "Processing.*files" output/ae_large_scale/version3_larger/logs/debug_rank*.log

# 查看是否有 worker 提前完成
grep "COMPLETED iteration" output/ae_large_scale/version3_larger/logs/debug_rank*.log

# 查看是否有读取错误
grep "ERROR.*shard" output/ae_large_scale/version3_larger/logs/debug_rank*.log
```

### 步骤 4：查看 batch 处理进度

```bash
# 查看训练 batch 进度（每 500 个 batch 记录一次）
grep "Processed.*training batches" output/ae_large_scale/version3_larger/logs/debug_rank0_*.log

# 查看验证 batch 进度（每 20 个 batch 记录一次）
grep "Validation step" output/ae_large_scale/version3_larger/logs/debug_rank0_*.log
```

## 📊 日志示例

### 正常的训练日志片段：

```
[INFO] Starting training...
[INFO] Monitor logs with: tail -f output/.../logs/train_rank0_20231203_143025.log
[DEBUG] [Rank 0/8 Worker 0/2] Processing 1000/8000 files from train_shards
[DEBUG] [Rank 0] ========== EPOCH 0 TRAINING START ==========
[DEBUG] [Rank 0] Training step 0, loss=0.1234
[DEBUG] [Rank 0] Training step 500, loss=0.0987
[DEBUG] [Rank 0] ========== EPOCH 0 TRAINING END ==========
[DEBUG] [Rank 0] ========== EPOCH 0 VALIDATION START ==========
[DEBUG] [Rank 0] Validation step 0, loss=0.1050
[DEBUG] [Rank 0] ========== EPOCH 0 VALIDATION END ==========
```

### 数据加载完成的标记：

```
[DEBUG] [Rank 0 Worker 0] COMPLETED iteration, total samples: 5000000
[DEBUG] [Rank 1 Worker 0] COMPLETED iteration, total samples: 5000000
[DEBUG] [Rank 2 Worker 0] COMPLETED iteration, total samples: 5000000
```

## ⚙️ 配置选项

### 关闭 debug 模式

编辑 `train_ae.sh`，修改：

```bash
--debug=true  # 改为 false 或删除这一行
```

关闭 debug 模式后：
- 不再记录详细的 shard 读取进度
- 不再记录每个 batch 的进度
- 日志文件会小得多
- 仅保留 INFO 级别的信息

### 调整日志频率

编辑 `train_ae.py` 的 `DebugCallback` 类：

```python
# 训练 batch 日志频率（默认每 500 个 batch）
if batch_idx % 500 == 0:  # 改为 100、1000 等

# 验证 batch 日志频率（默认每 20 个 batch）
if batch_idx % 20 == 0:   # 改为 10、50 等
```

## 💡 最佳实践

1. **训练前**：使用 `--debug=true` 启动，以便完整记录
2. **训练中**：在另一个终端用 `tail -f` 监控日志
3. **卡死时**：立即查看各个 rank 的 debug 日志最后几行
4. **训练后**：分析日志文件，查找性能瓶颈

## 🔧 故障排除

### 问题：日志文件没有生成

**解决**：检查 `$OUTPUT_DIR/logs` 目录权限

```bash
ls -la output/ae_large_scale/version3_larger/logs/
```

### 问题：日志文件太大

**解决**：
1. 关闭 debug 模式
2. 减少日志频率
3. 使用 `logrotate` 工具管理日志

### 问题：多个 rank 的日志混在一起

**解决**：每个 rank 会生成独立的 `debug_rank{N}_*.log` 文件，分别查看

## 📞 需要帮助？

如果遇到无法解决的卡死问题：

1. 收集所有日志文件
2. 记录卡死时的 GPU/CPU 使用情况（`nvidia-smi`, `htop`）
3. 记录最后 100 行的日志输出
