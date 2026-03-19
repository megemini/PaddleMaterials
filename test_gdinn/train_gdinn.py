#!/usr/bin/env python
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

"""
GDI-NN 训练脚本
基于 GDI-NN repo 的训练命令进行测试

原始命令：
python train.py \
    --model_type SolvGNN \
    --batch_size 1000 \
    --epochs 2 \
    --hidden_dim 64 \
    --lr 1e-3 \
    --pinn_lambda 1.0
"""

import argparse
import os
import sys
import paddle
import paddle.nn as nn
from omegaconf import OmegaConf

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ppmat'))

from ppmat.datasets import build_dataloader
from ppmat.models import SolvGNN
from ppmat.losses import GibbsDuhemLoss
from ppmat.metrics import MAE, RMSE, R2


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train GDI-NN model')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='SolvGNN',
                        help='Model type: SolvGNN')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension size')
    parser.add_argument('--pinn_lambda', type=float, default=1.0,
                        help='Gibbs-Duhem constraint loss weight')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    
    # 数据参数
    parser.add_argument('--train_data', type=str,
                        default='./data/gdinn/train_binary.csv',
                        help='Training data path')
    parser.add_argument('--val_data', type=str,
                        default='./data/gdinn/val_binary.csv',
                        help='Validation data path')
    
    # 其他参数
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--device', type=str, default='gpu',
                        help='Device: gpu or cpu')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log interval')
    
    return parser.parse_args()


def create_dataloader(data_path, solvent_list_path, batch_size, shuffle=True, num_workers=4):
    """创建数据加载器

    Args:
        data_path: Path to binary activity data CSV
        solvent_list_path: Path to solvent list CSV
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of workers
    """
    from ppmat.datasets import BinaryActivityDataset
    from paddle.io import DataLoader, BatchSampler

    # 创建数据集（BinaryActivityDataset 会自动检测并转换原始数据格式）
    dataset = BinaryActivityDataset(
        data_path=data_path,
        solvent_list_path=solvent_list_path,
        add_self_loop=True,
        preload_graphs=False
    )
    
    # 创建采样器
    sampler = BatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True
    )
    
    # 创建数据加载器
    from ppmat.datasets.collate_fn import DefaultCollator
    collator = DefaultCollator()
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=collator
    )
    
    return dataloader


def train(args):
    """训练模型"""
    print("=" * 80)
    print("GDI-NN 训练")
    print("=" * 80)
    print(f"模型类型: {args.model_type}")
    print(f"隐藏维度: {args.hidden_dim}")
    print(f"批次大小: {args.batch_size}")
    print(f"训练轮数: {args.epochs}")
    print(f"学习率: {args.lr}")
    print(f"PINN Lambda: {args.pinn_lambda}")
    print(f"设备: {args.device}")
    print("=" * 80)
    
    # 设置设备
    if args.device == 'gpu':
        paddle.set_device('gpu')
    else:
        paddle.set_device('cpu')
    
    # 创建数据加载器
    print("\n[1/5] 创建数据加载器...")
    solvent_list_path = './data/gdinn/solvent_list.csv'
    try:
        train_loader = create_dataloader(
            args.train_data,
            solvent_list_path,
            args.batch_size,
            shuffle=True,
            num_workers=4
        )
        val_loader = create_dataloader(
            args.val_data,
            solvent_list_path,
            args.batch_size,
            shuffle=False,
            num_workers=2
        )
        print(f"✓ 训练集 batch数: {len(train_loader)}")
        print(f"✓ 验证集 batch数: {len(val_loader)}")
    except Exception as e:
        print(f"✗ 数据加载器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 创建模型
    print("\n[2/5] 创建模型...")
    try:
        if args.model_type == 'SolvGNN':
            model = SolvGNN(
                in_dim=75,
                hidden_dim=args.hidden_dim,
                n_classes=1,
                mlp_dropout_rate=0.1,
                mlp_activation='softplus',
                mpnn_activation='relu',
                num_step_message_passing=6,
                pinn_lambda=args.pinn_lambda
            )
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")
        
        param_count = sum(p.numel().item() for p in model.parameters())
        print(f"✓ 模型创建成功")
        print(f"  参数数量: {param_count}")
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 创建损失函数
    print("\n[3/5] 创建损失函数...")
    try:
        criterion = GibbsDuhemLoss()
        print("✓ 损失函数创建成功")
    except Exception as e:
        print(f"✗ 损失函数创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 创建优化器
    print("\n[4/5] 创建优化器...")
    try:
        optimizer = paddle.optimizer.Adam(
            parameters=model.parameters(),
            learning_rate=args.lr,
            beta1=0.9,
            beta2=0.999,
            epsilon=1.0e-8,
            weight_decay=0.0
        )
        print(f"✓ 优化器创建成功")
        print(f"  学习率: {args.lr}")
    except Exception as e:
        print(f"✗ 优化器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 创建评估指标
    print("\n[5/5] 创建评估指标...")
    mae = MAE()
    rmse = RMSE()
    r2 = R2()
    print("✓ 评估指标创建成功 (MAE, RMSE, R2)")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 训练循环
    print("\n" + "=" * 80)
    print("开始训练")
    print("=" * 80)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch + 1}/{args.epochs}]")
        print("-" * 80)
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # 前向传播
            output = model(batch)
            loss_dict = output['loss_dict']
            pred_dict = output['pred_dict']
            
            # 计算损失
            loss = loss_dict['total_loss']
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            paddle.nn.ClipGradNorm(1.0)(model.parameters())
            
            # 参数更新
            optimizer.step()
            optimizer.clear_grad()
            
            train_loss += loss.item()
            train_steps += 1
            
            # 打印日志
            if (batch_idx + 1) % args.log_interval == 0:
                avg_loss = train_loss / train_steps
                print(f"  Batch [{batch_idx + 1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, "
                      f"Avg: {avg_loss:.4f}")
        
        avg_train_loss = train_loss / train_steps if train_steps > 0 else 0.0
        print(f"\n  训练损失: {avg_train_loss:.4f}")
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_steps = 0
        
        mae.reset()
        rmse.reset()
        r2.reset()
        
        with paddle.no_grad():
            for batch in val_loader:
                output = model(batch)
                loss_dict = output['loss_dict']
                pred_dict = output['pred_dict']
                
                loss = loss_dict['total_loss']
                val_loss += loss.item()
                val_steps += 1
                
                # 更新评估指标
                targets = paddle.stack([batch['gamma1'], batch['gamma2']], axis=1)
                predictions = paddle.stack([pred_dict['gamma1'], pred_dict['gamma2']], axis=1)
                
                mae.update(predictions, targets)
                rmse.update(predictions, targets)
                r2.update(predictions, targets)
        
        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0.0
        print(f"  验证损失: {avg_val_loss:.4f}")
        print(f"  MAE: {mae.accumulate():.4f}")
        print(f"  RMSE: {rmse.accumulate():.4f}")
        print(f"  R2: {r2.accumulate():.4f}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(args.output_dir, 'best_model.pdparams')
            paddle.save(model.state_dict(), save_path)
            print(f"  ✓ 保存最佳模型: {save_path}")
    
    # 保存最终模型
    final_save_path = os.path.join(args.output_dir, 'final_model.pdparams')
    paddle.save(model.state_dict(), final_save_path)
    
    print("\n" + "=" * 80)
    print("训练完成！")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"模型已保存到: {args.output_dir}")
    print("=" * 80)
    
    return True


def main():
    """主函数"""
    args = parse_args()
    
    try:
        success = train(args)
        if success:
            print("\n✓ 训练成功！")
            return 0
        else:
            print("\n✗ 训练失败！")
            return 1
    except Exception as e:
        print(f"\n✗ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
