# GDI-NN 快速开始指南

本文档介绍如何使用 GDI-NN 进行训练和预测。

## 测试流程

### 1. 快速测试（推荐首先运行）

首先运行快速测试脚本，验证所有组件是否正常工作：

```bash
python quick_test.py
```

这个脚本会：
- 自动创建测试数据
- 测试数据加载
- 测试模型前向传播
- 测试训练步骤
- 测试预测功能

如果所有测试通过，就可以进行完整的训练和预测。

### 2. 完整训练

使用与 GDI-NN repo 相同的命令格式进行训练：

```bash
python train_gdinn.py \
    --model_type SolvGNN \
    --batch_size 1000 \
    --epochs 2 \
    --hidden_dim 64 \
    --lr 1e-3 \
    --pinn_lambda 1.0
```

**参数说明：**
- `--model_type`: 模型类型，可选 `SolvGNN` 或 `SolvGNNWithHydrogenBonds`
- `--batch_size`: 批次大小
- `--epochs`: 训练轮数
- `--hidden_dim`: 隐藏层维度
- `--lr`: 学习率
- `--pinn_lambda`: PINN 约束损失权重

**其他可选参数：**
- `--train_data`: 训练数据路径（默认：`./data/gdinn/train_binary.csv`）
- `--val_data`: 验证数据路径（默认：`./data/gdinn/val_binary.csv`）
- `--output_dir`: 模型保存目录（默认：`./checkpoints`）
- `--device`: 设备类型（`gpu` 或 `cpu`，默认：`gpu`）
- `--log_interval`: 日志打印间隔（默认：10）

### 3. 预测

训练完成后，使用预测脚本进行预测：

```bash
python predict_gdinn.py \
    --model_type SolvGNN \
    --batch_size 1000 \
    --hidden_dim 64 \
    --checkpoint ./checkpoints/best_model.pdparams
```

**参数说明：**
- `--model_type`: 模型类型（必须与训练时一致）
- `--batch_size`: 批次大小
- `--hidden_dim`: 隐藏层维度（必须与训练时一致）
- `--checkpoint`: 模型检查点路径
- `--pinn_lambda`: PINN 约束损失权重（必须与训练时一致）

**其他可选参数：**
- `--test_data`: 测试数据路径（默认：`./data/gdinn/test_binary.csv`）
- `--output_file`: 预测结果保存路径（默认：`./predictions.csv`）
- `--device`: 设备类型（`gpu` 或 `cpu`，默认：`gpu`）

## 数据格式

训练、验证和测试数据应该是 CSV 格式，包含以下列：

- `solv1`: 溶剂1的 SMILES
- `solv2`: 溶剂2的 SMILES
- `T`: 温度（K）
- `x1`: 溶剂1的摩尔分数
- `x2`: 溶剂2的摩尔分数（x2 = 1 - x1）
- `gamma1`: 溶剂1的活度系数
- `gamma2`: 溶剂2的活度系数

示例：

```csv
solv1,solv2,T,x1,x2,gamma1,gamma2
O,CCO,298.15,0.5,0.5,1.2,0.8
CO,CCO,298.15,0.3,0.7,1.1,0.9
```

## 训练输出

训练过程中会显示：
- 每个 epoch 的训练损失
- 验证集上的损失和评估指标（MAE、RMSE、R2）
- 模型检查点保存信息

训练完成后，会在 `--output_dir` 目录下保存：
- `best_model.pdparams`: 验证集上表现最好的模型
- `final_model.pdparams`: 最后一个 epoch 的模型

## 预测输出

预测完成后会：
- 在控制台显示评估指标（MAE、RMSE、R2）
- 将预测结果保存到 `--output_file` 指定的 CSV 文件
- 显示前10个样本的预测结果

## 常见问题

### 1. CUDA out of memory

如果遇到 GPU 内存不足，可以：
- 减小 `--batch_size`
- 减小 `--hidden_dim`
- 使用 `--device cpu` 在 CPU 上运行

### 2. 数据加载错误

确保：
- 数据文件存在
- 数据格式正确（包含必需的列）
- SMILES 字符串有效

### 3. 模型不收敛

可以尝试：
- 调整学习率 `--lr`
- 增加 `--hidden_dim`
- 增加 `--epochs`
- 调整 `--pinn_lambda` 值

## 完整示例

```bash
# 1. 快速测试
python quick_test.py

# 2. 训练模型
python train_gdinn.py \
    --model_type SolvGNN \
    --batch_size 32 \
    --epochs 50 \
    --hidden_dim 128 \
    --lr 1e-3 \
    --pinn_lambda 1.0 \
    --output_dir ./my_checkpoints

# 3. 预测
python predict_gdinn.py \
    --model_type SolvGNN \
    --batch_size 32 \
    --hidden_dim 128 \
    --checkpoint ./my_checkpoints/best_model.pdparams \
    --output_file ./my_predictions.csv
```

## 与原始 GDI-NN repo 的对应关系

| 原始命令 | 新脚本 | 说明 |
|---------|--------|------|
| `python train.py` | `python train_gdinn.py` | 训练脚本 |
| `--model_type SolvGNN` | `--model_type SolvGNN` | 模型类型 |
| `--batch_size 1000` | `--batch_size 1000` | 批次大小 |
| `--epochs 2` | `--epochs 2` | 训练轮数 |
| `--hidden_dim 64` | `--hidden_dim 64` | 隐藏维度 |
| `--lr 1e-3` | `--lr 1e-3` | 学习率 |
| `--pinn_lambda 1.0` | `--pinn_lambda 1.0` | PINN 权重 |

主要改进：
1. 使用 PaddleMaterials 的标准数据加载器
2. 集成 BuildMolecule 工厂函数
3. 支持配置文件和命令行参数
4. 自动保存最佳模型
5. 完整的评估指标计算
6. 更详细的日志输出
