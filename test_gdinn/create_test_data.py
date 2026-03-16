#!/usr/bin/env python
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# 创建测试数据集

import csv
import os

def create_test_data():
    """创建测试数据集"""
    print("创建测试数据集...")
    
    # 创建测试数据目录
    os.makedirs('./data/gdinn', exist_ok=True)
    
    # 创建训练数据
    train_data = [
        ['solv1', 'solv2', 'solv1_x', 'gamma1', 'gamma2'],
        ['CCO', 'CC(C)O', '0.1', '1.2', '1.1'],
        ['CCO', 'CC(C)O', '0.3', '1.5', '1.3'],
        ['CCO', 'CC(C)O', '0.5', '1.8', '1.6'],
        ['CCO', 'CC(C)O', '0.7', '2.1', '1.9'],
        ['CCO', 'CC(C)O', '0.9', '2.5', '2.3'],
    ]
    
    with open('./data/gdinn/train_binary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(train_data)
    
    print(f"✓ 训练数据创建完成: ./data/gdinn/train_binary.csv ({len(train_data)-1} 样本)")
    
    # 创建验证数据
    val_data = [
        ['solv1', 'solv2', 'solv1_x', 'gamma1', 'gamma2'],
        ['CCO', 'CC(C)O', '0.2', '1.3', '1.2'],
        ['CCO', 'CC(C)O', '0.4', '1.6', '1.4'],
        ['CCO', 'CC(C)O', '0.6', '1.9', '1.7'],
        ['CCO', 'CC(C)O', '0.8', '2.2', '2.0'],
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
