#!/bin/bash

# GDI-NN 测试脚本
# 运行所有测试以验证 GDI-NN 集成

set -e

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 测试计数器
PASSED=0
FAILED=0

# 辅助函数
run_test() {
    local test_name=$1
    local test_command=$2
    
    echo "=========================================="
    echo "运行测试: $test_name"
    echo "=========================================="
    
    if eval $test_command; then
        echo -e "${GREEN}✓ $test_name 测试通过${NC}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}✗ $test_name 测试失败${NC}"
        FAILED=$((FAILED + 1))
    fi
    echo ""
}

# 打印开始信息
echo "=========================================="
echo "GDI-NN 测试套件"
echo "=========================================="
echo ""

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 1. 创建测试数据
echo -e "${YELLOW}[1/4] 创建测试数据...${NC}"
run_test "创建测试数据" "python test_gdinn/create_test_data.py"

# 2. 运行快速测试
echo -e "${YELLOW}[2/4] 运行快速测试...${NC}"
run_test "快速测试" "python test_gdinn/quick_test.py"

# 3. 运行训练测试
echo -e "${YELLOW}[3/4] 运行训练测试 (1 epoch)...${NC}"
run_test "训练测试" "python test_gdinn/train_gdinn.py --epochs 1 --batch_size 32 --hidden_dim 64 --lr 1e-3 --pinn_lambda 1.0 --output_dir ./test_output"

# 4. 运行预测测试
echo -e "${YELLOW}[4/4] 运行预测测试...${NC}"
if [ -f "./test_output/best_model.pdparams" ]; then
    run_test "预测测试" "python test_gdinn/predict_gdinn.py --checkpoint ./test_output/best_model.pdparams --batch_size 32 --hidden_dim 64 --output_file ./test_predictions.csv"
else
    echo -e "${YELLOW}⚠ 跳过预测测试 (模型不存在)${NC}"
    echo ""
fi

# 打印总结
echo "=========================================="
echo "测试总结"
echo "=========================================="
echo -e "通过: ${GREEN}${PASSED}${NC}"
echo -e "失败: ${RED}${FAILED}${NC}"
echo "总计: $((PASSED + FAILED))"
echo "=========================================="

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ 所有测试通过！${NC}"
    exit 0
else
    echo -e "${RED}✗ 部分测试失败${NC}"
    exit 1
fi
