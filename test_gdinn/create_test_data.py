#!/usr/bin/env python
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# 创建测试数据集

import csv
import os

def create_test_data():
    """创建测试数据集（使用真实数据格式）"""
    print("创建测试数据集...")

    # 创建测试数据目录
    os.makedirs('./data/gdinn', exist_ok=True)

    import numpy as np

    # 创建训练数据（真实数据格式）
    train_data = [
        ['SMILES_x', 'SMILES_y', 'temperature (K)', 'x(1)', 'x(2)', 'ln_gamma_1', 'ln_gamma_2'],
        ['CCO', 'CC(C)O', '298.15', '0.1', '0.9', f'{np.log(1.2):.6f}', f'{np.log(1.1):.6f}'],
        ['CCO', 'CC(C)O', '298.15', '0.3', '0.7', f'{np.log(1.5):.6f}', f'{np.log(1.3):.6f}'],
        ['CCO', 'CC(C)O', '298.15', '0.5', '0.5', f'{np.log(1.8):.6f}', f'{np.log(1.6):.6f}'],
        ['CCO', 'CC(C)O', '298.15', '0.7', '0.3', f'{np.log(2.1):.6f}', f'{np.log(1.9):.6f}'],
        ['CCO', 'CC(C)O', '298.15', '0.9', '0.1', f'{np.log(2.5):.6f}', f'{np.log(2.3):.6f}'],
        ['C', 'CCO', '298.15', '0.1', '0.9', f'{np.log(1.1):.6f}', f'{np.log(1.05):.6f}'],
        ['C', 'CCO', '298.15', '0.5', '0.5', f'{np.log(1.4):.6f}', f'{np.log(1.3):.6f}'],
        ['C', 'CCO', '298.15', '0.9', '0.1', f'{np.log(1.7):.6f}', f'{np.log(1.5):.6f}'],
    ]

    with open('./data/gdinn/train_binary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(train_data)

    print(f"✓ 训练数据创建完成: ./data/gdinn/train_binary.csv ({len(train_data)-1} 样本)")

    # 创建验证数据
    val_data = [
        ['SMILES_x', 'SMILES_y', 'temperature (K)', 'x(1)', 'x(2)', 'ln_gamma_1', 'ln_gamma_2'],
        ['CCO', 'CC(C)O', '298.15', '0.2', '0.8', f'{np.log(1.3):.6f}', f'{np.log(1.2):.6f}'],
        ['CCO', 'CC(C)O', '298.15', '0.4', '0.6', f'{np.log(1.6):.6f}', f'{np.log(1.4):.6f}'],
        ['CCO', 'CC(C)O', '298.15', '0.6', '0.4', f'{np.log(1.9):.6f}', f'{np.log(1.7):.6f}'],
        ['CCO', 'CC(C)O', '298.15', '0.8', '0.2', f'{np.log(2.2):.6f}', f'{np.log(2.0):.6f}'],
        ['C', 'CCO', '298.15', '0.3', '0.7', f'{np.log(1.25):.6f}', f'{np.log(1.15):.6f}'],
        ['C', 'CCO', '298.15', '0.7', '0.3', f'{np.log(1.55):.6f}', f'{np.log(1.35):.6f}'],
    ]

    with open('./data/gdinn/val_binary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(val_data)

    print(f"✓ 验证数据创建完成: ./data/gdinn/val_binary.csv ({len(val_data)-1} 样本)")

    # 创建测试数据
    with open('./data/gdinn/test_binary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(val_data)

    print(f"✓ 测试数据创建完成: ./data/gdinn/test_binary.csv ({len(val_data)-1} 样本)")
    print("\n测试数据创建完成！")

if __name__ == "__main__":
    create_test_data()
