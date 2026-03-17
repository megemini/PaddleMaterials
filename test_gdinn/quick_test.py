#!/usr/bin/env python
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

"""
GDI-NN 快速测试脚本
验证训练和预测流程是否正常工作
"""

import os
import sys
import paddle
import numpy as np
import pandas as pd

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ppmat'))


def create_test_data():
    """创建测试数据（使用 GDI-NN 格式）"""
    import pandas as pd

    print("准备测试数据（使用 GDI-NN 格式）...")

    # 路径配置
    dataset_dir = './test_gdinn/dataset'
    solvent_list_path = os.path.join(dataset_dir, 'solvent_list.csv')
    # 使用 GDI-NN 格式的数据文件
    output_binary_path = os.path.join(dataset_dir, 'output_binary_with_inf_all copy.csv')

    # 创建输出目录
    output_dir = './data/gdinn'
    os.makedirs(output_dir, exist_ok=True)

    # 复制溶剂列表（直接使用）
    print(f"读取溶剂列表: {solvent_list_path}")
    solvent_df = pd.read_csv(solvent_list_path)
    solvent_df.to_csv(os.path.join(output_dir, 'solvent_list.csv'), index=False)
    print(f"✓ 溶剂数量: {len(solvent_df)}")

    # 读取 GDI-NN 格式数据
    print(f"读取数据: {output_binary_path}")
    df = pd.read_csv(output_binary_path)
    print(f"✓ 数据量: {len(df)}")

    # GDI-NN 格式：solv1_gamma, solv2_gamma 存储 ln_gamma
    # 过滤极端值
    df = df[~df['solv1_gamma'].isna() & ~df['solv2_gamma'].isna()]
    df = df[(abs(df['solv1_gamma']) <= 50) & (abs(df['solv2_gamma']) <= 50)]

    # 取前5000条数据进行测试（如果数据量足够）
    df = df.head(5000)

    # 分割数据集（使用比例：80% 训练，10% 验证，10% 测试）
    n = len(df)
    train_ratio = 0.8
    val_ratio = 0.1

    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]

    # 保存数据（GDI-NN 格式）
    train_df.to_csv(os.path.join(output_dir, 'train_binary.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_binary.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_binary.csv'), index=False)

    print(f"✓ 训练集: {len(train_df)} 样本")
    print(f"✓ 验证集: {len(val_df)} 样本")
    print(f"✓ 测试集: {len(test_df)} 样本")
    print(f"✓ 数据保存在: {output_dir}/")


def test_data_loading():
    """测试数据加载"""
    print("\n" + "=" * 80)
    print("测试数据加载")
    print("=" * 80)

    try:
        from ppmat.datasets import BinaryActivityDataset
        from paddle.io import DataLoader, BatchSampler
        from ppmat.datasets.collate_fn import DefaultCollator

        # 创建数据集（GDI-NN 格式）
        dataset = BinaryActivityDataset(
            data_path='./data/gdinn/train_binary.csv',
            solvent_list_path='./data/gdinn/solvent_list.csv',
            add_self_loop=True,
            preload_graphs=False,
            compute_hb=False
        )
        
        print(f"✓ 数据集创建成功")
        print(f"  样本数量: {len(dataset)}")
        
        # 创建采样器
        sampler = BatchSampler(
            dataset=dataset,
            batch_size=32,
            shuffle=True,
            drop_last=True
        )
        
        # 创建数据加载器
        collator = DefaultCollator()
        dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=collator
        )
        
        print(f"✓ 数据加载器创建成功")
        print(f"  Batch数量: {len(dataloader)}")
        
        # 测试加载数据
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 2:  # 只测试前2个batch
                break
            
            print(f"\nBatch {batch_idx + 1}:")
            print(f"  g1 nodes: {batch['g1'].num_nodes}")
            print(f"  g1 edges: {batch['g1'].num_edges}")
            print(f"  g2 nodes: {batch['g2'].num_nodes}")
            print(f"  g2 edges: {batch['g2'].num_edges}")
            if 'T' in batch:
                print(f"  T shape: {batch['T'].shape}")
            print(f"  x1 shape: {batch['x1'].shape}")
            print(f"  x2 shape: {batch['x2'].shape}")
            print(f"  gamma1 shape: {batch['gamma1'].shape}")
            print(f"  gamma2 shape: {batch['gamma2'].shape}")
        
        print("\n✓ 数据加载测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_forward():
    """测试模型前向传播"""
    print("\n" + "=" * 80)
    print("测试模型前向传播")
    print("=" * 80)
    
    try:
        from ppmat.models import SolvGNN
        from ppmat.datasets import BinaryActivityDataset
        from paddle.io import DataLoader, BatchSampler
        from ppmat.datasets.collate_fn import DefaultCollator
        
        # 创建模型 (使用与原始 GDI-NN 一致的默认参数)
        model = SolvGNN(
            in_dim=75,
            hidden_dim=64,
            n_classes=1,
            num_step_message_passing=1,
            pinn_lambda=1.0
        )

        print(f"✓ 模型创建成功")
        param_count = sum(p.numel().item() for p in model.parameters())
        print(f"  参数数量: {param_count}")

        # 创建数据加载器（GDI-NN 格式）
        dataset = BinaryActivityDataset(
            data_path='./data/gdinn/train_binary.csv',
            solvent_list_path='./data/gdinn/solvent_list.csv',
            add_self_loop=True,
            preload_graphs=False,
            compute_hb=False
        )

        sampler = BatchSampler(
            dataset=dataset,
            batch_size=32,
            shuffle=False,
            drop_last=True
        )

        collator = DefaultCollator()
        dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=collator
        )
        
        # 测试前向传播
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 1:
                break
            
            print(f"\n测试 Batch {batch_idx + 1}...")
            
            # 前向传播
            output = model(batch)
            
            print(f"✓ 前向传播成功")
            print(f"  loss_dict keys: {list(output['loss_dict'].keys())}")
            print(f"  pred_dict keys: {list(output['pred_dict'].keys())}")
            print(f"  total_loss: {output['loss_dict']['total_loss'].item():.4f}")
            print(f"  pred_loss: {output['loss_dict'].get('pred_loss', 0).item():.4f}")
            print(f"  gd_loss: {output['loss_dict'].get('gd_loss', 0).item():.4f}")
            print(f"  gamma1 shape: {output['pred_dict']['gamma1'].shape}")
            print(f"  gamma2 shape: {output['pred_dict']['gamma2'].shape}")
        
        print("\n✓ 模型前向传播测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 模型前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step():
    """测试训练步骤"""
    print("\n" + "=" * 80)
    print("测试训练步骤")
    print("=" * 80)
    
    try:
        from ppmat.models import SolvGNN
        from ppmat.losses import GDICombinedLoss
        from ppmat.datasets import BinaryActivityDataset
        from paddle.io import DataLoader, BatchSampler
        from ppmat.datasets.collate_fn import DefaultCollator
        
        # 创建模型 (使用与原始 GDI-NN 一致的默认参数)
        model = SolvGNN(
            in_dim=75,
            hidden_dim=64,
            n_classes=1,
            num_step_message_passing=1,
            pinn_lambda=1.0
        )

        # 创建损失函数
        criterion = GDICombinedLoss(
            lambda_gd=1.0,
            gd_start_epoch=5,
            loss_type='mse',
            gd_loss_type='mse',
            use_ln_gamma=True
        )

        print(f"✓ 模型和损失函数创建成功")

        # 创建数据加载器（GDI-NN 格式）
        dataset = BinaryActivityDataset(
            data_path='./data/gdinn/train_binary.csv',
            solvent_list_path='./data/gdinn/solvent_list.csv',
            add_self_loop=True,
            preload_graphs=False,
            compute_hb=False
        )

        sampler = BatchSampler(
            dataset=dataset,
            batch_size=32,
            shuffle=True,
            drop_last=True
        )
        
        collator = DefaultCollator()
        dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=collator
        )
        
        # 创建优化器
        optimizer = paddle.optimizer.Adam(
            parameters=model.parameters(),
            learning_rate=0.001
        )
        
        print(f"✓ 优化器创建成功")
        
        # 测试训练步骤
        model.train()
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 3:
                break
            
            # 前向传播
            output = model(batch)
            loss = output['loss_dict']['total_loss']
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            paddle.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 参数更新
            optimizer.step()
            optimizer.clear_grad()
            
            print(f"  Step {batch_idx + 1}: Loss = {loss.item():.4f}")
        
        print("\n✓ 训练步骤测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 训练步骤测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_solvgnn_xmlp_forward():
    """测试 SolvGNNxMLP 模型前向传播"""
    print("\n" + "=" * 80)
    print("测试 SolvGNNxMLP 模型前向传播")
    print("=" * 80)
    
    try:
        from ppmat.models import SolvGNNxMLP
        from ppmat.datasets import BinaryActivityDataset
        from paddle.io import DataLoader, BatchSampler
        from ppmat.datasets.collate_fn import DefaultCollator
        
        # 创建模型
        model = SolvGNNxMLP(
            in_dim=75,
            hidden_dim=64,
            n_classes=1,
            mlp_num_hid_layers=2,
            num_step_message_passing=1,
            pinn_lambda=1.0
        )

        print(f"✓ SolvGNNxMLP 模型创建成功")
        param_count = sum(p.numel().item() for p in model.parameters())
        print(f"  参数数量: {param_count}")

        # 创建数据加载器（GDI-NN 格式）
        dataset = BinaryActivityDataset(
            data_path='./data/gdinn/train_binary.csv',
            solvent_list_path='./data/gdinn/solvent_list.csv',
            add_self_loop=True,
            preload_graphs=False,
            compute_hb=False
        )

        sampler = BatchSampler(
            dataset=dataset,
            batch_size=32,
            shuffle=False,
            drop_last=True
        )

        collator = DefaultCollator()
        dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=collator
        )
        
        # 测试前向传播
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 1:
                break
            
            print(f"\n测试 Batch {batch_idx + 1}...")
            
            # 前向传播
            output = model(batch)
            
            print(f"✓ 前向传播成功")
            print(f"  loss_dict keys: {list(output['loss_dict'].keys())}")
            print(f"  pred_dict keys: {list(output['pred_dict'].keys())}")
            print(f"  total_loss: {output['loss_dict']['total_loss'].item():.4f}")
            print(f"  pred_loss: {output['loss_dict'].get('pred_loss', 0).item():.4f}")
            print(f"  gd_loss: {output['loss_dict'].get('gd_loss', 0).item():.4f}")
            print(f"  gamma1 shape: {output['pred_dict']['gamma1'].shape}")
            print(f"  gamma2 shape: {output['pred_dict']['gamma2'].shape}")
        
        print("\n✓ SolvGNNxMLP 模型前向传播测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ SolvGNNxMLP 模型前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_solvgnn_xmlp_training():
    """测试 SolvGNNxMLP 模型训练步骤"""
    print("\n" + "=" * 80)
    print("测试 SolvGNNxMLP 模型训练步骤")
    print("=" * 80)
    
    try:
        from ppmat.models import SolvGNNxMLP
        from ppmat.datasets import BinaryActivityDataset
        from paddle.io import DataLoader, BatchSampler
        from ppmat.datasets.collate_fn import DefaultCollator
        
        # 创建模型
        model = SolvGNNxMLP(
            in_dim=75,
            hidden_dim=64,
            n_classes=1,
            mlp_num_hid_layers=2,
            num_step_message_passing=1,
            pinn_lambda=1.0
        )

        print(f"✓ SolvGNNxMLP 模型创建成功")

        # 创建数据加载器（GDI-NN 格式）
        dataset = BinaryActivityDataset(
            data_path='./data/gdinn/train_binary.csv',
            solvent_list_path='./data/gdinn/solvent_list.csv',
            add_self_loop=True,
            preload_graphs=False,
            compute_hb=False
        )

        sampler = BatchSampler(
            dataset=dataset,
            batch_size=32,
            shuffle=True,
            drop_last=True
        )

        collator = DefaultCollator()
        dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=collator
        )
        
        # 创建优化器
        optimizer = paddle.optimizer.Adam(
            parameters=model.parameters(),
            learning_rate=0.001
        )
        
        print(f"✓ 优化器创建成功")
        
        # 测试训练步骤
        model.train()
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 3:
                break
            
            # 前向传播
            output = model(batch)
            loss = output['loss_dict']['total_loss']
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            paddle.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 参数更新
            optimizer.step()
            optimizer.clear_grad()
            
            print(f"  Step {batch_idx + 1}: Loss = {loss.item():.4f}")
        
        print("\n✓ SolvGNNxMLP 训练步骤测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ SolvGNNxMLP 训练步骤测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gegnn_forward():
    """测试 GEGNN 模型前向传播"""
    print("\n" + "=" * 80)
    print("测试 GEGNN 模型前向传播")
    print("=" * 80)
    
    try:
        from ppmat.models import GEGNN
        from ppmat.datasets import BinaryActivityDataset
        from paddle.io import DataLoader, BatchSampler
        from ppmat.datasets.collate_fn import DefaultCollator
        
        # 创建模型
        model = GEGNN(
            in_dim=75,
            hidden_dim=64,
            n_classes=1,
            num_step_message_passing=1,
            pinn_lambda=1.0
        )

        print(f"✓ GEGNN 模型创建成功")
        param_count = sum(p.numel().item() for p in model.parameters())
        print(f"  参数数量: {param_count}")

        # 创建数据加载器（GDI-NN 格式）
        dataset = BinaryActivityDataset(
            data_path='./data/gdinn/train_binary.csv',
            solvent_list_path='./data/gdinn/solvent_list.csv',
            add_self_loop=True,
            preload_graphs=False,
            compute_hb=False
        )

        sampler = BatchSampler(
            dataset=dataset,
            batch_size=32,
            shuffle=False,
            drop_last=True
        )

        collator = DefaultCollator()
        dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=collator
        )
        
        # 测试前向传播
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 1:
                break
            
            print(f"\n测试 Batch {batch_idx + 1}...")
            
            # 前向传播
            output = model(batch)
            
            print(f"✓ 前向传播成功")
            print(f"  loss_dict keys: {list(output['loss_dict'].keys())}")
            print(f"  pred_dict keys: {list(output['pred_dict'].keys())}")
            print(f"  total_loss: {output['loss_dict']['total_loss'].item():.4f}")
            print(f"  pred_loss: {output['loss_dict'].get('pred_loss', 0).item():.4f}")
            print(f"  gd_loss: {output['loss_dict'].get('gd_loss', 0).item():.4f}")
            print(f"  gamma1 shape: {output['pred_dict']['gamma1'].shape}")
            print(f"  gamma2 shape: {output['pred_dict']['gamma2'].shape}")
            if 'G_E' in output['pred_dict']:
                print(f"  G_E shape: {output['pred_dict']['G_E'].shape}")
                print(f"  G_E mean: {output['pred_dict']['G_E'].mean().item():.4f}")
        
        print("\n✓ GEGNN 模型前向传播测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ GEGNN 模型前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gegnn_training():
    """测试 GEGNN 模型训练步骤"""
    print("\n" + "=" * 80)
    print("测试 GEGNN 模型训练步骤")
    print("=" * 80)
    
    try:
        from ppmat.models import GEGNN
        from ppmat.datasets import BinaryActivityDataset
        from paddle.io import DataLoader, BatchSampler
        from ppmat.datasets.collate_fn import DefaultCollator
        
        # 创建模型
        model = GEGNN(
            in_dim=75,
            hidden_dim=64,
            n_classes=1,
            num_step_message_passing=1,
            pinn_lambda=1.0
        )

        print(f"✓ GEGNN 模型创建成功")

        # 创建数据加载器（GDI-NN 格式）
        dataset = BinaryActivityDataset(
            data_path='./data/gdinn/train_binary.csv',
            solvent_list_path='./data/gdinn/solvent_list.csv',
            add_self_loop=True,
            preload_graphs=False,
            compute_hb=False
        )

        sampler = BatchSampler(
            dataset=dataset,
            batch_size=32,
            shuffle=True,
            drop_last=True
        )

        collator = DefaultCollator()
        dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=collator
        )

        # 创建优化器
        optimizer = paddle.optimizer.Adam(
            parameters=model.parameters(),
            learning_rate=0.001
        )
        
        print(f"✓ 优化器创建成功")
        
        # 测试训练步骤
        model.train()
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 3:
                break
            
            # 前向传播
            output = model(batch)
            loss = output['loss_dict']['total_loss']
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            paddle.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 参数更新
            optimizer.step()
            optimizer.clear_grad()
            
            print(f"  Step {batch_idx + 1}: Loss = {loss.item():.4f}, "
                  f"G_E = {output['pred_dict']['G_E'].mean().item():.4f}")
        
        print("\n✓ GEGNN 训练步骤测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ GEGNN 训练步骤测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_test_data_mcm():
    """创建 MCM 模型测试数据（使用 ID 而非 SMILES）"""
    import pandas as pd
    
    print("创建 MCM 测试数据...")
    
    # 创建简单的测试数据
    data = []
    
    # 使用 ID 而非 SMILES
    # 假设有 10 种不同的溶剂
    solvent_pairs = [
        # Solvent ID pairs (solv1_id, solv2_id, temp, x1, gamma1, gamma2)
        (0, 1, 298.15, 0.5, 1.2, 0.8),
        (1, 2, 298.15, 0.5, 1.1, 0.9),
        (0, 2, 298.15, 0.5, 1.3, 0.7),
        (2, 3, 298.15, 0.5, 1.15, 0.85),
        (3, 4, 298.15, 0.5, 1.25, 0.75),
    ]
    
    # 重复生成更多数据
    for solv1_id, solv2_id, temp, x1, gamma1, gamma2 in solvent_pairs:
        for _ in range(100):  # 每个组合生成100个样本
            # 添加一些随机变化
            x1_var = np.clip(x1 + np.random.normal(0, 0.1), 0.01, 0.99)
            x2 = 1.0 - x1_var
            
            # 简单的活度系数模拟
            gamma1_var = gamma1 * (1 + 0.1 * np.random.randn())
            gamma2_var = gamma2 * (1 + 0.1 * np.random.randn())
            
            # 转换为 ln_gamma
            ln_gamma1_var = np.log(abs(gamma1_var))
            ln_gamma2_var = np.log(abs(gamma2_var))
            
            data.append({
                'solv1_id': solv1_id,
                'solv2_id': solv2_id,
                'temperature (K)': temp + np.random.normal(0, 5),
                'x(1)': x1_var,
                'x(2)': x2,
                'ln_gamma_1': ln_gamma1_var,
                'ln_gamma_2': ln_gamma2_var
            })
    
    # 创建目录
    os.makedirs('./data/gdinn', exist_ok=True)
    
    # 保存数据
    df = pd.DataFrame(data)
    
    # 分割数据集
    train_df = df.iloc[:400]
    val_df = df.iloc[400:450]
    test_df = df.iloc[450:]
    
    train_df.to_csv('./data/gdinn/train_mcm.csv', index=False)
    val_df.to_csv('./data/gdinn/val_mcm.csv', index=False)
    test_df.to_csv('./data/gdinn/test_mcm.csv', index=False)
    
    print(f"✓ MCM 训练集: {len(train_df)} 样本")
    print(f"✓ MCM 验证集: {len(val_df)} 样本")
    print(f"✓ MCM 测试集: {len(test_df)} 样本")
    print(f"✓ 数据保存在: ./data/gdinn/")
    
    return 5  # 返回最大 solvent ID


def test_mcm_forward():
    """测试 MCM 模型前向传播"""
    print("\n" + "=" * 80)
    print("测试 MCM 模型前向传播")
    print("=" * 80)
    
    try:
        from ppmat.models.gdinn.mcm import MCM_MultiMLP
        import pandas as pd
        
        # 创建测试数据
        max_solvent_id = create_test_data_mcm()
        
        # 创建模型
        model = MCM_MultiMLP(
            solvent_id_max=max_solvent_id,
            dim_hidden_channels=64,
            dropout_hidden=0.05,
            dropout_interaction=0.03,
            mlp_num_hid_layers=1,
            pinn_lambda=1.0
        )
        
        print(f"✓ MCM 模型创建成功")
        param_count = sum(p.numel().item() for p in model.parameters())
        print(f"  参数数量: {param_count}")
        
        # 加载测试数据
        df = pd.read_csv('./data/gdinn/train_mcm.csv')
        
        # 创建 batch data
        batch_size = 32
        batch_data = {
            'solv1_id': paddle.to_tensor(df['solv1_id'].values[:batch_size], dtype='int64'),
            'solv2_id': paddle.to_tensor(df['solv2_id'].values[:batch_size], dtype='int64'),
            'x1': paddle.to_tensor(df['x(1)'].values[:batch_size], dtype='float32'),
            'gamma1': paddle.to_tensor(df['ln_gamma_1'].values[:batch_size], dtype='float32').unsqueeze(-1),
            'gamma2': paddle.to_tensor(df['ln_gamma_2'].values[:batch_size], dtype='float32').unsqueeze(-1)
        }
        
        print(f"\n测试前向传播...")
        
        # 前向传播
        output = model(batch_data)
        
        print(f"✓ 前向传播成功")
        print(f"  loss_dict keys: {list(output['loss_dict'].keys())}")
        print(f"  pred_dict keys: {list(output['pred_dict'].keys())}")
        print(f"  total_loss: {output['loss_dict']['total_loss'].item():.4f}")
        print(f"  pred_loss: {output['loss_dict'].get('pred_loss', 0).item():.4f}")
        print(f"  gd_loss: {output['loss_dict'].get('gd_loss', 0).item():.4f}")
        print(f"  gamma1 shape: {output['pred_dict']['gamma1'].shape}")
        print(f"  gamma2 shape: {output['pred_dict']['gamma2'].shape}")
        
        print("\n✓ MCM 模型前向传播测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ MCM 模型前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mcm_training():
    """测试 MCM 模型训练步骤"""
    print("\n" + "=" * 80)
    print("测试 MCM 模型训练步骤")
    print("=" * 80)
    
    try:
        from ppmat.models.gdinn.mcm import MCM_MultiMLP
        import pandas as pd
        
        # 创建测试数据
        max_solvent_id = 5
        
        # 创建模型
        model = MCM_MultiMLP(
            solvent_id_max=max_solvent_id,
            dim_hidden_channels=64,
            dropout_hidden=0.05,
            dropout_interaction=0.03,
            mlp_num_hid_layers=1,
            pinn_lambda=1.0
        )
        
        print(f"✓ MCM 模型创建成功")
        
        # 加载训练数据
        df = pd.read_csv('./data/gdinn/train_mcm.csv')
        
        # 创建优化器
        optimizer = paddle.optimizer.Adam(
            parameters=model.parameters(),
            learning_rate=0.001
        )
        
        print(f"✓ 优化器创建成功")
        
        # 测试训练步骤
        model.train()
        
        batch_size = 32
        num_batches = min(3, len(df) // batch_size)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            # 创建 batch data
            batch_data = {
                'solv1_id': paddle.to_tensor(df['solv1_id'].values[start_idx:end_idx], dtype='int64'),
                'solv2_id': paddle.to_tensor(df['solv2_id'].values[start_idx:end_idx], dtype='int64'),
                'x1': paddle.to_tensor(df['x(1)'].values[start_idx:end_idx], dtype='float32'),
                'gamma1': paddle.to_tensor(df['ln_gamma_1'].values[start_idx:end_idx], dtype='float32').unsqueeze(-1),
                'gamma2': paddle.to_tensor(df['ln_gamma_2'].values[start_idx:end_idx], dtype='float32').unsqueeze(-1)
            }
            
            # 前向传播
            output = model(batch_data)
            loss = output['loss_dict']['total_loss']
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            paddle.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 参数更新
            optimizer.step()
            optimizer.clear_grad()
            
            print(f"  Step {batch_idx + 1}: Loss = {loss.item():.4f}")
        
        print("\n✓ MCM 训练步骤测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ MCM 训练步骤测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prediction():
    """测试预测"""
    print("\n" + "=" * 80)
    print("测试预测")
    print("=" * 80)
    
    try:
        from ppmat.models import SolvGNN
        from ppmat.datasets import BinaryActivityDataset
        from paddle.io import DataLoader, BatchSampler
        from ppmat.datasets.collate_fn import DefaultCollator
        
        # 创建模型 (使用与原始 GDI-NN 一致的默认参数)
        model = SolvGNN(
            in_dim=75,
            hidden_dim=64,
            n_classes=1,
            num_step_message_passing=1,
            pinn_lambda=1.0
        )

        # 创建测试数据加载器（GDI-NN 格式）
        dataset = BinaryActivityDataset(
            data_path='./data/gdinn/test_binary.csv',
            solvent_list_path='./data/gdinn/solvent_list.csv',
            add_self_loop=True,
            preload_graphs=False,
            compute_hb=False
        )

        sampler = BatchSampler(
            dataset=dataset,
            batch_size=32,
            shuffle=False,
            drop_last=False
        )

        collator = DefaultCollator()
        dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=collator
        )

        # 测试预测
        model.eval()
        
        all_predictions = []
        
        with paddle.no_grad():
            for batch in dataloader:
                output = model(batch)
                pred_dict = output['pred_dict']
                
                gamma1_pred = pred_dict['gamma1'].numpy()
                gamma2_pred = pred_dict['gamma2'].numpy()
                gamma1_target = batch['gamma1'].numpy()
                gamma2_target = batch['gamma2'].numpy()
                
                for i in range(len(gamma1_pred)):
                    all_predictions.append({
                        'gamma1_pred': float(gamma1_pred[i][0]),
                        'gamma2_pred': float(gamma2_pred[i][0]),
                        'gamma1_target': float(gamma1_target[i][0]),
                        'gamma2_target': float(gamma2_target[i][0])
                    })
        
        print(f"✓ 预测完成")
        print(f"  预测样本数: {len(all_predictions)}")
        
        # 计算简单的误差
        mae = np.mean([
            abs(p['gamma1_pred'] - p['gamma1_target']) +
            abs(p['gamma2_pred'] - p['gamma2_target'])
            for p in all_predictions
        ]) / 2
        
        print(f"  平均绝对误差 (MAE): {mae:.4f}")
        
        # 显示前5个预测结果
        print(f"\n前5个预测结果:")
        for i, pred in enumerate(all_predictions[:5]):
            print(f"  {i+1}. gamma1: pred={pred['gamma1_pred']:.4f}, target={pred['gamma1_target']:.4f}, "
                  f"error={abs(pred['gamma1_pred'] - pred['gamma1_target']):.4f}")
            print(f"     gamma2: pred={pred['gamma2_pred']:.4f}, target={pred['gamma2_target']:.4f}, "
                  f"error={abs(pred['gamma2_pred'] - pred['gamma2_target']):.4f}")
        
        print("\n✓ 预测测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 预测测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("=" * 80)
    print("GDI-NN 快速测试")
    print("=" * 80)
    
    # 创建测试数据
    create_test_data()
    
    # 运行测试
    results = {}
    
    results['数据加载'] = test_data_loading()
    results['SolvGNN前向传播'] = test_model_forward()
    results['SolvGNN训练步骤'] = test_training_step()
    results['SolvGNNxMLP前向传播'] = test_solvgnn_xmlp_forward()
    results['SolvGNNxMLP训练步骤'] = test_solvgnn_xmlp_training()
    results['GEGNN前向传播'] = test_gegnn_forward()
    results['GEGNN训练步骤'] = test_gegnn_training()
    results['MCM前向传播'] = test_mcm_forward()
    results['MCM训练步骤'] = test_mcm_training()
    results['预测'] = test_prediction()
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    print("=" * 80)
    
    if passed == total:
        print("✓ 所有测试通过！")
        print("\n现在可以使用以下命令进行完整训练:")
        print("  python train_gdinn.py \\")
        print("    --model_type SolvGNN \\")
        print("    --batch_size 32 \\")
        print("    --epochs 2 \\")
        print("    --hidden_dim 64 \\")
        print("    --lr 1e-3 \\")
        print("    --pinn_lambda 1.0")
        print("\n训练完成后，可以使用以下命令进行预测:")
        print("  python predict_gdinn.py \\")
        print("    --model_type SolvGNN \\")
        print("    --batch_size 32 \\")
        print("    --hidden_dim 64 \\")
        print("    --checkpoint ./checkpoints/best_model.pdparams")
        return 0
    else:
        print("✗ 部分测试失败，请检查错误信息")
        return 1


if __name__ == "__main__":
    sys.exit(main())
