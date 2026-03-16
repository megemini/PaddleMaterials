# GDI-NN 测试目录

本目录包含 GDI-NN 模型的训练、预测和测试脚本。

## 文件说明

### 核心脚本

1. **`train_gdinn.py`** - 训练脚本
   - 训练 GDI-NN 模型
   - 支持命令行参数配置
   - 自动保存最佳模型

2. **`predict_gdinn.py`** - 预测脚本
   - 使用训练好的模型进行预测
   - 计算评估指标
   - 保存预测结果

3. **`quick_test.py`** - 快速测试脚本
   - 自动创建测试数据
   - 测试数据加载
   - 测试模型前向传播
   - 测试训练步骤
   - 测试预测功能

4. **`create_test_data.py`** - 创建测试数据
   - 生成训练/验证/测试数据
   - 数据保存在 `./data/gdinn/` 目录

5. **`run_all_tests.sh`** - 运行所有测试
   - 一键运行所有测试
   - 自动化测试流程

### 文档

- **`QUICK_START_GUIDE.md`** - 快速开始指南
  - 详细的训练和预测说明
  - 参数说明
  - 常见问题解答

## 快速开始

### 1. 运行所有测试

```bash
cd /home/shun/workspace/Projects/megemini/PaddleMaterials
./test_gdinn/run_all_tests.sh
```

### 2. 快速测试

```bash
cd /home/shun/workspace/Projects/megemini/PaddleMaterials
python test_gdinn/quick_test.py
```

### 3. 训练模型

```bash
cd /home/shun/workspace/Projects/megemini/PaddleMaterials
python test_gdinn/train_gdinn.py \
    --model_type SolvGNN \
    --batch_size 1000 \
    --epochs 2 \
    --hidden_dim 64 \
    --lr 1e-3 \
    --pinn_lambda 1.0
```

### 4. 预测

```bash
cd /home/shun/workspace/Projects/megemini/PaddleMaterials
python test_gdinn/predict_gdinn.py \
    --model_type SolvGNN \
    --batch_size 1000 \
    --hidden_dim 64 \
    --checkpoint ./checkpoints/best_model.pdparams
```

## 测试流程

1. **创建测试数据** → `create_test_data.py`
2. **快速测试** → `quick_test.py`
3. **训练** → `train_gdinn.py`
4. **预测** → `predict_gdinn.py`

或使用一键测试：
```bash
./run_all_tests.sh
```

## 参数说明

### 训练参数

- `--model_type`: 模型类型（`SolvGNN` 或 `SolvGNNWithHydrogenBonds`）
- `--batch_size`: 批次大小
- `--epochs`: 训练轮数
- `--hidden_dim`: 隐藏层维度
- `--lr`: 学习率
- `--pinn_lambda`: PINN 约束损失权重
- `--train_data`: 训练数据路径
- `--val_data`: 验证数据路径
- `--output_dir`: 模型保存目录
- `--device`: 设备（`gpu` 或 `cpu`）

### 预测参数

- `--model_type`: 模型类型
- `--batch_size`: 批次大小
- `--hidden_dim`: 隐藏层维度
- `--checkpoint`: 模型检查点路径
- `--test_data`: 测试数据路径
- `--output_file`: 预测结果保存路径
- `--device`: 设备

## 输出说明

### 训练输出

- 模型保存在 `./checkpoints/` 目录
- `best_model.pdparams`: 验证集上表现最好的模型
- `final_model.pdparams`: 最后一个 epoch 的模型

### 预测输出

- 预测结果保存到 `./predictions.csv`
- 包含预测值和真实值
- 控制台显示评估指标（MAE、RMSE、R2）

## 常见问题

### CUDA out of memory

减小 `--batch_size` 或使用 `--device cpu`

### 数据加载错误

确保数据文件存在且格式正确

### 模型不收敛

调整学习率、隐藏层维度或训练轮数

## 详细文档

查看 `QUICK_START_GUIDE.md` 获取更详细的说明和示例。
