#!/usr/bin/env python
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

"""
GDI-NN 精度对齐测试脚本

验证 PaddleMaterials 的 gdinn 实现与原始 GDI-NN (PyTorch) 实现的模型结构一致性:
- ppmat/models/gdinn/gnn.py <-> model/model_GNN.py
- ppmat/models/gdinn/mcm.py <-> model/model_MCM.py

通过对比相同随机输入的输出来验证模型结构一致性
"""

import os
import sys
import numpy as np
import pandas as pd
import paddle
import torch
import dgl

# 设置 Paddle 使用 GPU
paddle.set_device('gpu:0')

# 设置 PyTorch 使用 GPU (GDI-NN 代码内部硬编码了 .cuda())
torch.cuda.set_device(0)

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, '/home/shun/workspace/Projects/github/GDI-NN')

# ============================================================================
# 配置路径 (参考 quick_test.py)
# ============================================================================
GDI_NN_DIR = '/home/shun/workspace/Projects/github/GDI-NN'

class Config:
    # 数据目录配置
    DATASET_DIR = './test_gdinn/dataset'
    OUTPUT_DIR = './test_gdinn/data/alignment_test'

    # 数据文件
    SOLVENT_LIST_FILE = 'solvent_list.csv'
    BINARY_DATA_FILE = 'output_binary_with_inf_all.csv'
    
    # 精度阈值
    FORWARD_TOLERANCE = 1e-3
    
    # 测试参数
    BATCH_SIZE = 8
    HIDDEN_DIM = 64
    IN_DIM = 75
    NUM_CLASSES = 1

    @property
    def solvent_list_path(self):
        return os.path.join(self.DATASET_DIR, self.SOLVENT_LIST_FILE)

    @property
    def binary_data_path(self):
        return os.path.join(self.DATASET_DIR, self.BINARY_DATA_FILE)
    
    @property
    def output_dir(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        return self.OUTPUT_DIR


config = Config()


def set_random_seed(seed=42):
    """设置随机种子以确保可重复性"""
    np.random.seed(seed)
    paddle.seed(seed)
    torch.manual_seed(seed)


def prepare_data():
    """准备真实测试数据"""
    print("准备测试数据...")
    
    # 读取数据
    solvent_df = pd.read_csv(config.solvent_list_path)
    df = pd.read_csv(config.binary_data_path)
    
    # 过滤
    df = df[~df['solv1_gamma'].isna() & ~df['solv2_gamma'].isna()]
    df = df[(abs(df['solv1_gamma']) <= 50) & (abs(df['solv2_gamma']) <= 50)]
    
    # 取前500条数据进行测试
    df = df.head(500)
    
    print(f"✓ 加载数据: {len(df)} 样本, {len(solvent_df)} 溶剂")
    
    return df, solvent_df


# ============================================================================
# GNN 模型精度对齐测试
# ============================================================================

def test_gnn_alignment():
    """测试 GNN 模型精度对齐 (Paddle vs PyTorch)"""
    print("\n" + "=" * 80)
    print("测试 GNN 精度对齐 (Paddle vs PyTorch)")
    print("=" * 80)
    
    try:
        # 准备数据
        df, solvent_df = prepare_data()
        
        # 导入 Paddle 模型和数据集
        from ppmat.models.gdinn.gnn import SolvGNN
        from ppmat.datasets import BinaryActivityDataset
        from paddle.io import DataLoader, BatchSampler
        from ppmat.datasets.collate_fn import DefaultCollator
        
        # 导入 PyTorch 模型
        sys.path.insert(0, GDI_NN_DIR)
        from model.model_GNN import solvgnn_binary
        
        # 创建 Paddle 数据集
        paddle_dataset = BinaryActivityDataset(
            data_path=config.binary_data_path,
            solvent_list_path=config.solvent_list_path,
            add_self_loop=True,
            preload_graphs=False,
            compute_hb=False
        )
        
        # 创建采样器
        sampler = BatchSampler(
            dataset=paddle_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            drop_last=False
        )
        
        collator = DefaultCollator()
        paddle_loader = DataLoader(
            dataset=paddle_dataset,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=collator
        )
        
        # 创建 Paddle 模型
        paddle_model = SolvGNN(
            in_dim=config.IN_DIM,
            hidden_dim=config.HIDDEN_DIM,
            n_classes=config.NUM_CLASSES,
            num_step_message_passing=1,
            pinn_lambda=0.0
        )
        paddle_model.eval()
        
        # 创建 PyTorch 模型并移到 GPU
        torch_model = solvgnn_binary(
            in_dim=config.IN_DIM,
            hidden_dim=config.HIDDEN_DIM,
            n_classes=config.NUM_CLASSES,
            mlp_dropout_rate=0,
            mlp_activation="relu",
            mpnn_activation="relu",
            mlp_num_hid_layers=2
        )
        torch_model = torch_model.cuda()
        torch_model.eval()
        
        # 获取一个 batch
        for batch_idx, paddle_batch in enumerate(paddle_loader):
            if batch_idx >= 1:
                break
            
            # 先获取 PyTorch 数据 (在 Paddle 前向传播之前，避免图特征被修改)
            g1_paddle = paddle_batch['g1']
            g2_paddle = paddle_batch['g2']
            x1_np = paddle_batch['x1'].numpy()
            
            # 转换 Paddle 图到 DGL 图
            g1_dgl = paddle_graph_to_dgl(g1_paddle)
            g2_dgl = paddle_graph_to_dgl(g2_paddle)
            
            # 创建 empty solvsys
            num_nodes = 2 * config.BATCH_SIZE
            empty_solvsys = create_empty_solvsys(num_nodes)
            
            # PyTorch 前向传播 (solv1_x 需要 1D tensor)
            torch_batch = {
                'g1': g1_dgl,
                'g2': g2_dgl,
                'solv1_x': torch.from_numpy(x1_np).flatten(),
                'inter_hb': torch.zeros(config.BATCH_SIZE),
                'intra_hb1': torch.zeros(config.BATCH_SIZE),
                'intra_hb2': torch.zeros(config.BATCH_SIZE),
            }
            
            with torch.no_grad():
                torch_output = torch_model(torch_batch, empty_solvsys, gamma_grad=False)
            
            torch_gamma1 = torch_output[:, 0].numpy()
            torch_gamma2 = torch_output[:, 1].numpy()
            
            # Paddle 前向传播
            with paddle.no_grad():
                paddle_output = paddle_model(paddle_batch)
            
            paddle_pred = paddle_output['pred_dict']
            paddle_gamma1 = paddle_pred['gamma1'].numpy()
            paddle_gamma2 = paddle_pred['gamma2'].numpy()
            
            # 比较
            diff_gamma1 = np.abs(paddle_gamma1.flatten() - torch_gamma1)
            diff_gamma2 = np.abs(paddle_gamma2.flatten() - torch_gamma2)
            
            max_diff_gamma1 = np.max(diff_gamma1)
            max_diff_gamma2 = np.max(diff_gamma2)
            mean_diff_gamma1 = np.mean(diff_gamma1)
            mean_diff_gamma2 = np.mean(diff_gamma2)
            
            print(f"  gamma1 最大差异: {max_diff_gamma1:.6f}")
            print(f"  gamma1 平均差异: {mean_diff_gamma1:.6f}")
            print(f"  gamma2 最大差异: {max_diff_gamma2:.6f}")
            print(f"  gamma2 平均差异: {mean_diff_gamma2:.6f}")
            
            passed = max(max_diff_gamma1, max_diff_gamma2) < config.FORWARD_TOLERANCE
            
            if passed:
                return True, "精度对齐"
            else:
                return False, f"gamma1 max diff: {max_diff_gamma1:.6f}, gamma2 max diff: {max_diff_gamma2:.6f}"
        
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False, f"导入失败: {e}"
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


def paddle_graph_to_dgl(paddle_g):
    """将 Paddle batched 图转换为 DGL batched 图"""
    # 获取节点和边信息
    src = paddle_g.edges[0].numpy() if hasattr(paddle_g.edges[0], 'numpy') else paddle_g.edges[0]
    dst = paddle_g.edges[1].numpy() if hasattr(paddle_g.edges[1], 'numpy') else paddle_g.edges[1]
    num_nodes = paddle_g.num_nodes
    
    # 获取节点特征 (Paddle 使用 node_feat 字典)
    if paddle_g.node_feat and 'h' in paddle_g.node_feat:
        node_feats = paddle_g.node_feat['h'].numpy()
    elif paddle_g.node_feat and 'feat' in paddle_g.node_feat:
        node_feats = paddle_g.node_feat['feat'].numpy()
    else:
        node_feats = np.random.randn(num_nodes, config.IN_DIM).astype(np.float32)
    
    # 创建 DGL 图并移到 GPU
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    g = g.to("cuda:0")  # 先把图移到 GPU
    g.ndata['h'] = torch.from_numpy(node_feats).cuda()
    
    # 设置 batch 信息 (如果存在)
    if hasattr(paddle_g, 'batch_num_nodes') and paddle_g.batch_num_nodes is not None:
        g.set_batch_num_nodes(paddle_g.batch_num_nodes.numpy() if hasattr(paddle_g.batch_num_nodes, 'numpy') else paddle_g.batch_num_nodes)
    if hasattr(paddle_g, 'batch_num_edges') and paddle_g.batch_num_edges is not None:
        g.set_batch_num_edges(paddle_g.batch_num_edges.numpy() if hasattr(paddle_g.batch_num_edges, 'numpy') else paddle_g.batch_num_edges)
    
    return g


def create_empty_solvsys(num_nodes):
    """创建空的溶剂系统图 (GPU)"""
    g = dgl.graph(([], []), num_nodes=num_nodes)
    return g.to("cuda:0")


# ============================================================================
# MCM 模型精度对齐测试
# ============================================================================

def test_mcm_alignment():
    """测试 MCM 模型精度对齐 (Paddle vs PyTorch)"""
    print("\n" + "=" * 80)
    print("测试 MCM 精度对齐 (Paddle vs PyTorch)")
    print("=" * 80)
    
    try:
        # 准备数据
        df, solvent_df = prepare_data()
        
        # 导入 Paddle MCM 模型
        from ppmat.models.gdinn.mcm import MCM_MultiMLP
        
        # 导入 PyTorch MCM 模型
        sys.path.insert(0, GDI_NN_DIR)
        from model.model_MCM import MCM_multiMLP
        
        # 创建 Paddle MCM 模型
        solvent_id_max = len(solvent_df)
        
        paddle_model = MCM_MultiMLP(
            solvent_id_max=solvent_id_max,
            dim_hidden_channels=config.HIDDEN_DIM,
            dropout_hidden=0.0,
            dropout_interaction=0.0,
            mlp_num_hid_layers=1,
            pinn_lambda=0.0
        )
        paddle_model.eval()
        
        # 创建 PyTorch MCM 模型并移到 GPU
        torch_model = MCM_multiMLP(
            solvent_id_max=solvent_id_max,
            dim_hidden_channels=config.HIDDEN_DIM,
            dropout_hidden=0.0,
            dropout_interaction=0.0,
            mlp_num_hid_layers=1
        )
        torch_model = torch_model.cuda()
        torch_model.eval()
        
        # 准备测试数据 (从真实数据中取)
        batch_size = min(config.BATCH_SIZE, len(df))
        
        # 查找溶剂 ID
        solv1_smiles = df['solv1'].values[:batch_size]
        solv2_smiles = df['solv2'].values[:batch_size]
        
        # 创建 SMILES 到 ID 的映射
        solvent_list = solvent_df['smiles_can'].tolist()
        solv1_id = [solvent_list.index(s) if s in solvent_list else 0 for s in solv1_smiles]
        solv2_id = [solvent_list.index(s) if s in solvent_list else 0 for s in solv2_smiles]
        
        x1 = df['solv1_x'].values[:batch_size].astype(np.float32)
        gamma1 = df['solv1_gamma'].values[:batch_size].astype(np.float32)
        gamma2 = df['solv2_gamma'].values[:batch_size].astype(np.float32)
        
        # Paddle 前向传播
        paddle_batch = {
            'solv1_id': paddle.to_tensor(solv1_id, dtype='int64'),
            'solv2_id': paddle.to_tensor(solv2_id, dtype='int64'),
            'x1': paddle.to_tensor(x1, dtype='float32'),
            'gamma1': paddle.to_tensor(gamma1, dtype='float32').reshape([-1, 1]),
            'gamma2': paddle.to_tensor(gamma2, dtype='float32').reshape([-1, 1]),
        }
        
        with paddle.no_grad():
            paddle_output = paddle_model(paddle_batch)
        
        paddle_pred = paddle_output['pred_dict']
        paddle_ln_gamma1 = paddle_pred['ln_gamma1'].numpy()
        paddle_ln_gamma2 = paddle_pred['ln_gamma2'].numpy()
        
        # PyTorch 前向传播 (solv1_x 需要 1D tensor)
        torch_batch = {
            'solv1_id': torch.tensor(solv1_id, dtype=torch.int64),
            'solv2_id': torch.tensor(solv2_id, dtype=torch.int64),
            'solv1_x': torch.tensor(x1, dtype=torch.float32).flatten(),
            'gamma1': torch.tensor(gamma1, dtype=torch.float32).reshape([-1, 1]),
            'gamma2': torch.tensor(gamma2, dtype=torch.float32).reshape([-1, 1]),
        }
        
        with torch.no_grad():
            torch_output = torch_model(torch_batch, None, gamma_grad=False)
        
        torch_ln_gamma1 = torch_output[:, 0].cpu().numpy()
        torch_ln_gamma2 = torch_output[:, 1].cpu().numpy()
        
        # 比较
        diff_ln_gamma1 = np.abs(paddle_ln_gamma1.flatten() - torch_ln_gamma1)
        diff_ln_gamma2 = np.abs(paddle_ln_gamma2.flatten() - torch_ln_gamma2)
        
        max_diff_gamma1 = np.max(diff_ln_gamma1)
        max_diff_gamma2 = np.max(diff_ln_gamma2)
        mean_diff_gamma1 = np.mean(diff_ln_gamma1)
        mean_diff_gamma2 = np.mean(diff_ln_gamma2)
        
        print(f"  ln_gamma1 最大差异: {max_diff_gamma1:.6f}")
        print(f"  ln_gamma1 平均差异: {mean_diff_gamma1:.6f}")
        print(f"  ln_gamma2 最大差异: {max_diff_gamma2:.6f}")
        print(f"  ln_gamma2 平均差异: {mean_diff_gamma2:.6f}")
        
        passed = max(max_diff_gamma1, max_diff_gamma2) < config.FORWARD_TOLERANCE
        
        if passed:
            return True, "精度对齐"
        else:
            return False, f"ln_gamma1 max diff: {max_diff_gamma1:.6f}, ln_gamma2 max diff: {max_diff_gamma2:.6f}"
        
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False, f"导入失败: {e}"
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("=" * 80)
    print("GDI-NN 精度对齐测试 (使用真实数据)")
    print("=" * 80)
    
    # 设置随机种子
    set_random_seed(42)
    
    # 创建输出目录
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # 运行测试
    results = {}
    
    # GNN 精度对齐测试
    results['GNN_精度对齐'] = test_gnn_alignment()
    
    # MCM 精度对齐测试
    results['MCM_精度对齐'] = test_mcm_alignment()
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for test_name, (result, message) in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status} ({message})")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {passed}/{len(results)} 测试通过")
    print("=" * 80)
    
    if failed == 0:
        print("✓ 所有测试通过!")
        return 0
    else:
        print(f"✗ {failed} 个测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
