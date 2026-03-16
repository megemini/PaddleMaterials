#!/usr/bin/env python
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

"""
数据验证脚本
验证 GDI-NN 训练数据的格式是否正确
"""

import argparse
import pandas as pd
import sys
import os


def validate_smiles(smiles):
    """验证 SMILES 字符串是否有效"""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def validate_data_file(file_path):
    """验证单个数据文件"""
    print(f"\n{'=' * 80}")
    print(f"验证文件: {file_path}")
    print(f"{'=' * 80}")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"✗ 文件不存在")
        return False
    
    # 读取文件
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"✗ 无法读取 CSV 文件: {e}")
        return False
    
    print(f"✓ 文件读取成功")
    print(f"  数据行数: {len(df)}")
    print(f"  数据列数: {len(df.columns)}")
    print(f"  列名: {list(df.columns)}")
    
    # 检查必需的列（真实数据格式）
    required_columns = ['SMILES_x', 'SMILES_y', 'temperature (K)', 'x(1)', 'x(2)', 'ln_gamma_1', 'ln_gamma_2']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"✗ 缺少必需的列: {missing_columns}")
        return False
    
    print(f"✓ 所有必需的列都存在")
    
    # 检查缺失值
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"⚠ 发现缺失值:")
        for col, count in null_counts.items():
            if count > 0:
                print(f"  {col}: {count} 个缺失值")
    else:
        print(f"✓ 没有缺失值")
    
    # 检查数据类型
    print(f"\n数据类型:")
    print(f"  SMILES_x: {df['SMILES_x'].dtype}")
    print(f"  SMILES_y: {df['SMILES_y'].dtype}")
    print(f"  temperature (K): {df['temperature (K)'].dtype}")
    print(f"  x(1): {df['x(1)'].dtype}")
    print(f"  x(2): {df['x(2)'].dtype}")
    print(f"  ln_gamma_1: {df['ln_gamma_1'].dtype}")
    print(f"  ln_gamma_2: {df['ln_gamma_2'].dtype}")

    # 检查数据范围
    print(f"\n数据范围:")
    print(f"  temperature (K): {df['temperature (K)'].min():.2f} - {df['temperature (K)'].max():.2f} K")
    print(f"  x(1): {df['x(1)'].min():.4f} - {df['x(1)'].max():.4f}")
    print(f"  x(2): {df['x(2)'].min():.4f} - {df['x(2)'].max():.4f}")
    print(f"  ln_gamma_1: {df['ln_gamma_1'].min():.4f} - {df['ln_gamma_1'].max():.4f}")
    print(f"  ln_gamma_2: {df['ln_gamma_2'].min():.4f} - {df['ln_gamma_2'].max():.4f}")

    # 检查 x(1) + x(2) 是否等于 1
    x_sum_error = abs(df['x(1)'] + df['x(2)'] - 1)
    invalid_rows = df[x_sum_error > 0.01]

    if len(invalid_rows) > 0:
        print(f"\n⚠ 警告: {len(invalid_rows)} 行的 x(1) + x(2) 不等于 1（误差 > 0.01）")
        print(f"  最大误差: {x_sum_error.max():.6f}")
        print(f"  示例:")
        print(f"    {invalid_rows[['x(1)', 'x(2)']].head(3)}")
    else:
        print(f"\n✓ x(1) + x(2) = 1（误差 < 0.01）")

    # 检查 ln_gamma 是否在合理范围内
    print(f"\n验证 ln_gamma 值...")
    extreme_ln_gamma1 = df[abs(df['ln_gamma_1']) > 20]
    extreme_ln_gamma2 = df[abs(df['ln_gamma_2']) > 20]

    if len(extreme_ln_gamma1) > 0:
        print(f"⚠ 警告: {len(extreme_ln_gamma1)} 行的 ln_gamma_1 绝对值 > 20")
    else:
        print(f"✓ ln_gamma_1 在合理范围内")

    if len(extreme_ln_gamma2) > 0:
        print(f"⚠ 警告: {len(extreme_ln_gamma2)} 行的 ln_gamma_2 绝对值 > 20")
    else:
        print(f"✓ ln_gamma_2 在合理范围内")

    # 检查 SMILES 有效性
    print(f"\n验证 SMILES 字符串...")
    invalid_smiles_x = []
    invalid_smiles_y = []

    for idx, row in df.iterrows():
        if not validate_smiles(row['SMILES_x']):
            invalid_smiles_x.append(idx)
        if not validate_smiles(row['SMILES_y']):
            invalid_smiles_y.append(idx)

    if invalid_smiles_x:
        print(f"⚠ {len(invalid_smiles_x)} 行的 SMILES_x 无效")
        print(f"  示例: {df.loc[invalid_smiles_x[:3], 'SMILES_x'].tolist()}")
    else:
        print(f"✓ 所有 SMILES_x 都有效")

    if invalid_smiles_y:
        print(f"⚠ {len(invalid_smiles_y)} 行的 SMILES_y 无效")
        print(f"  示例: {df.loc[invalid_smiles_y[:3], 'SMILES_y'].tolist()}")
    else:
        print(f"✓ 所有 SMILES_y 都有效")

    # 统计唯一溶剂
    print(f"\n统计信息:")
    unique_smiles_x = df['SMILES_x'].nunique()
    unique_smiles_y = df['SMILES_y'].nunique()
    print(f"  唯一 SMILES_x 数量: {unique_smiles_x}")
    print(f"  唯一 SMILES_y 数量: {unique_smiles_y}")

    # 检查重复数据
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"\n⚠ 发现 {duplicates} 行重复数据")
    else:
        print(f"\n✓ 没有重复数据")

    # 显示前几行数据
    print(f"\n前 3 行数据:")
    print(df.head(3).to_string())

    # 总结
    has_issues = (
        len(invalid_smiles_x) > 0 or
        len(invalid_smiles_y) > 0 or
        len(extreme_ln_gamma1) > 0 or
        len(extreme_ln_gamma2) > 0 or
        len(invalid_rows) > 0 or
        null_counts.sum() > 0
    )

    if not has_issues:
        print(f"\n{'=' * 80}")
        print(f"✓ 数据验证通过！")
        print(f"{'=' * 80}")
        return True
    else:
        print(f"\n{'=' * 80}")
        print(f"⚠ 数据验证完成，发现一些问题，建议修复后再使用")
        print(f"{'=' * 80}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Validate GDI-NN training data')
    parser.add_argument('--train_data', type=str,
                        default='test_gdinn/dataset/train_binary.csv',
                        help='Training data path')
    parser.add_argument('--val_data', type=str,
                        default='test_gdinn/dataset/val_binary.csv',
                        help='Validation data path')
    parser.add_argument('--test_data', type=str,
                        default='test_gdinn/dataset/test_binary.csv',
                        help='Test data path')
    parser.add_argument('--all', action='store_true',
                        help='Validate all data files')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GDI-NN 数据验证工具")
    print("=" * 80)
    
    if args.all:
        # 验证所有文件
        files = [args.train_data, args.val_data, args.test_data]
        results = []
        for file_path in files:
            if os.path.exists(file_path):
                result = validate_data_file(file_path)
                results.append(result)
            else:
                print(f"\n⚠ 跳过不存在的文件: {file_path}")
                results.append(None)
        
        # 总结
        print(f"\n{'=' * 80}")
        print("验证总结")
        print(f"{'=' * 80}")
        valid_count = sum(1 for r in results if r is True)
        invalid_count = sum(1 for r in results if r is False)
        skip_count = sum(1 for r in results if r is None)
        
        print(f"通过: {valid_count}")
        print(f"有问题: {invalid_count}")
        print(f"跳过: {skip_count}")
        
        if invalid_count == 0 and skip_count == 0:
            print(f"\n✓ 所有文件验证通过！")
            return 0
        else:
            print(f"\n⚠ 部分文件有问题或不存在")
            return 1
    else:
        # 只验证训练数据
        result = validate_data_file(args.train_data)
        return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
