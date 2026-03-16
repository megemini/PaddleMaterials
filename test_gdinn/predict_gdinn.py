#!/usr/bin/env python
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

"""
GDI-NN 预测脚本
"""

import argparse
import os
import sys
import paddle
import numpy as np
import pandas as pd

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ppmat'))

from ppmat.datasets import build_dataloader
from ppmat.models import SolvGNN


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Predict with GDI-NN model')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='SolvGNN',
                        help='Model type: SolvGNN or SolvGNNWithHydrogenBonds')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension size')
    parser.add_argument('--pinn_lambda', type=float, default=1.0,
                        help='Gibbs-Duhem constraint loss weight')
    
    # 预测参数
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Batch size')
    
    # 数据参数
    parser.add_argument('--test_data', type=str,
                        default='./data/gdinn/test_binary.csv',
                        help='Test data path')
    parser.add_argument('--output_file', type=str,
                        default='./predictions.csv',
                        help='Output file path')
    
    # 模型加载参数
    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoints/best_model.pdparams',
                        help='Model checkpoint path')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='gpu',
                        help='Device: gpu or cpu')
    
    return parser.parse_args()


def create_dataloader(data_path, batch_size, shuffle=False, num_workers=2):
    """创建数据加载器"""
    from ppmat.datasets import BinaryActivityDataset
    from paddle.io import DataLoader, BatchSampler
    
    # 创建数据集
    dataset = BinaryActivityDataset(
        data_path=data_path,
        add_self_loop=True,
        preload_graphs=False,
        compute_hb=False
    )
    
    # 创建采样器
    sampler = BatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False
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


def predict(args):
    """执行预测"""
    print("=" * 80)
    print("GDI-NN 预测")
    print("=" * 80)
    print(f"模型类型: {args.model_type}")
    print(f"隐藏维度: {args.hidden_dim}")
    print(f"批次大小: {args.batch_size}")
    print(f"设备: {args.device}")
    print(f"检查点: {args.checkpoint}")
    print(f"测试数据: {args.test_data}")
    print(f"输出文件: {args.output_file}")
    print("=" * 80)
    
    # 设置设备
    if args.device == 'gpu':
        paddle.set_device('gpu')
    else:
        paddle.set_device('cpu')
    
    # 创建数据加载器
    print("\n[1/4] 创建数据加载器...")
    try:
        test_loader = create_dataloader(
            args.test_data,
            args.batch_size,
            shuffle=False,
            num_workers=2
        )
        print(f"✓ 测试集 batch数: {len(test_loader)}")
    except Exception as e:
        print(f"✗ 数据加载器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 创建模型
    print("\n[2/4] 创建模型...")
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
        elif args.model_type == 'SolvGNNWithHydrogenBonds':
            from ppmat.models import SolvGNNWithHydrogenBonds
            model = SolvGNNWithHydrogenBonds(
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
        
        print("✓ 模型创建成功")
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 加载模型参数
    print("\n[3/4] 加载模型参数...")
    try:
        if not os.path.exists(args.checkpoint):
            print(f"✗ 模型检查点不存在: {args.checkpoint}")
            return False
        
        model_state_dict = paddle.load(args.checkpoint)
        model.set_state_dict(model_state_dict)
        print(f"✓ 模型参数加载成功")
    except Exception as e:
        print(f"✗ 模型参数加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 执行预测
    print("\n[4/4] 执行预测...")
    try:
        model.eval()
        
        all_predictions = []
        all_targets = []
        
        with paddle.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # 前向传播
                output = model(batch)
                pred_dict = output['pred_dict']
                
                # 获取预测值
                gamma1_pred = pred_dict['gamma1'].numpy()
                gamma2_pred = pred_dict['gamma2'].numpy()
                
                # 获取真实值
                gamma1_target = batch['gamma1'].numpy()
                gamma2_target = batch['gamma2'].numpy()
                
                # 收集预测结果
                batch_size = gamma1_pred.shape[0]
                for i in range(batch_size):
                    all_predictions.append({
                        'gamma1_pred': float(gamma1_pred[i][0]),
                        'gamma2_pred': float(gamma2_pred[i][0]),
                        'gamma1_target': float(gamma1_target[i][0]),
                        'gamma2_target': float(gamma2_target[i][0])
                    })
                    all_targets.append([
                        float(gamma1_target[i][0]),
                        float(gamma2_target[i][0])
                    ])
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  已处理 batch [{batch_idx + 1}/{len(test_loader)}]")
        
        # 计算评估指标
        from ppmat.metrics import MAE, RMSE, R2
        mae = MAE()
        rmse = RMSE()
        r2 = R2()
        
        predictions = paddle.to_tensor([[p['gamma1_pred'], p['gamma2_pred']] for p in all_predictions])
        targets = paddle.to_tensor(all_targets)
        
        mae.update(predictions, targets)
        rmse.update(predictions, targets)
        r2.update(predictions, targets)
        
        print("\n" + "=" * 80)
        print("预测完成！")
        print("=" * 80)
        print(f"样本数量: {len(all_predictions)}")
        print(f"MAE: {mae.accumulate():.4f}")
        print(f"RMSE: {rmse.accumulate():.4f}")
        print(f"R2: {r2.accumulate():.4f}")
        
        # 保存预测结果
        df = pd.DataFrame(all_predictions)
        df.to_csv(args.output_file, index=False)
        print(f"预测结果已保存到: {args.output_file}")
        
        # 显示前10个样本
        print("\n前10个样本的预测结果:")
        print(df.head(10).to_string())
        
        return True
        
    except Exception as e:
        print(f"✗ 预测失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    args = parse_args()
    
    try:
        success = predict(args)
        if success:
            print("\n✓ 预测成功！")
            return 0
        else:
            print("\n✗ 预测失败！")
            return 1
    except Exception as e:
        print(f"\n✗ 预测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
