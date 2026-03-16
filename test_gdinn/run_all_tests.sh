#!/bin/bash

# GDI-NN 测试脚本
# 支持使用真实数据或测试数据

set -e

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# 解析命令行参数
USE_REAL_DATA=false
RUN_VALIDATION=false
RUN_QUICK_TEST=false
RUN_TRAINING=false
RUN_PREDICTION=false

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --real-data)
            USE_REAL_DATA=true
            shift
            ;;
        --validate)
            RUN_VALIDATION=true
            shift
            ;;
        --quick-test)
            RUN_QUICK_TEST=true
            shift
            ;;
        --train)
            RUN_TRAINING=true
            shift
            ;;
        --predict)
            RUN_PREDICTION=true
            shift
            ;;
        --all)
            RUN_VALIDATION=true
            RUN_QUICK_TEST=true
            RUN_TRAINING=true
            RUN_PREDICTION=true
            shift
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --real-data      使用真实数据（test_gdinn/dataset/）"
            echo "  --validate       验证数据格式"
            echo "  --quick-test     运行快速测试"
            echo "  --train          运行训练测试"
            echo "  --predict        运行预测测试"
            echo "  --all            运行所有测试（默认）"
            echo "  --help           显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0 --validate              # 只验证数据"
            echo "  $0 --real-data --train     # 使用真实数据训练"
            echo "  $0 --all                   # 运行所有测试（使用测试数据）"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 如果没有指定任何选项，默认运行所有测试
if [ "$RUN_VALIDATION" = false ] && [ "$RUN_QUICK_TEST" = false ] && [ "$RUN_TRAINING" = false ] && [ "$RUN_PREDICTION" = false ]; then
    RUN_VALIDATION=true
    RUN_QUICK_TEST=true
    RUN_TRAINING=true
    RUN_PREDICTION=true
fi

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 设置数据路径
if [ "$USE_REAL_DATA" = true ]; then
    TRAIN_DATA="test_gdinn/dataset/train_binary.csv"
    VAL_DATA="test_gdinn/dataset/val_binary.csv"
    TEST_DATA="test_gdinn/dataset/test_binary.csv"
    echo -e "${BLUE}使用真实数据: test_gdinn/dataset/${NC}"
else
    TRAIN_DATA="data/gdinn/train_binary.csv"
    VAL_DATA="data/gdinn/val_binary.csv"
    TEST_DATA="data/gdinn/test_binary.csv"
    echo -e "${BLUE}使用测试数据: data/gdinn/${NC}"
fi
echo ""

# 1. 数据验证（如果使用真实数据）
if [ "$RUN_VALIDATION" = true ] && [ "$USE_REAL_DATA" = true ]; then
    echo -e "${YELLOW}[1/4] 验证真实数据...${NC}"
    run_test "数据验证" "python test_gdinn/validate_data.py --all --train_data $TRAIN_DATA --val_data $VAL_DATA --test_data $TEST_DATA"
elif [ "$RUN_VALIDATION" = true ]; then
    echo -e "${YELLOW}[1/4] 创建测试数据...${NC}"
    run_test "创建测试数据" "python test_gdinn/create_test_data.py"
fi

# 2. 运行快速测试
if [ "$RUN_QUICK_TEST" = true ]; then
    if [ "$USE_REAL_DATA" = false ]; then
        echo -e "${YELLOW}[2/4] 运行快速测试...${NC}"
        run_test "快速测试" "python test_gdinn/quick_test.py"
    else
        echo -e "${YELLOW}[2/4] 跳过快速测试（使用真实数据时不需要）${NC}"
        echo ""
    fi
fi

# 3. 运行训练测试
if [ "$RUN_TRAINING" = true ]; then
    echo -e "${YELLOW}[3/4] 运行训练测试 (1 epoch)...${NC}"
    
    # 检查数据文件是否存在
    if [ "$USE_REAL_DATA" = true ] && [ ! -f "$TRAIN_DATA" ]; then
        echo -e "${RED}✗ 训练数据不存在: $TRAIN_DATA${NC}"
        echo "请确保真实数据文件已放置在 test_gdinn/dataset/ 目录下"
        FAILED=$((FAILED + 1))
        echo ""
    else
        run_test "训练测试" "python test_gdinn/train_gdinn.py --epochs 1 --batch_size 32 --hidden_dim 64 --lr 1e-3 --pinn_lambda 1.0 --train_data $TRAIN_DATA --val_data $VAL_DATA --output_dir ./test_output"
    fi
fi

# 4. 运行预测测试
if [ "$RUN_PREDICTION" = true ]; then
    echo -e "${YELLOW}[4/4] 运行预测测试...${NC}"
    
    if [ -f "./test_output/best_model.pdparams" ]; then
        run_test "预测测试" "python test_gdinn/predict_gdinn.py --checkpoint ./test_output/best_model.pdparams --batch_size 32 --hidden_dim 64 --test_data $TEST_DATA --output_file ./test_predictions.csv"
    else
        echo -e "${YELLOW}⚠ 跳过预测测试 (模型不存在)${NC}"
        echo ""
    fi
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
    
    if [ "$USE_REAL_DATA" = true ]; then
        echo ""
        echo -e "${BLUE}使用真实数据训练和预测示例：${NC}"
        echo ""
        echo "训练："
        echo "  python test_gdinn/train_gdinn.py \\"
        echo "    --model_type SolvGNN \\"
        echo "    --batch_size 1000 \\"
        echo "    --epochs 50 \\"
        echo "    --hidden_dim 128 \\"
        echo "    --lr 1e-3 \\"
        echo "    --pinn_lambda 1.0 \\"
        echo "    --train_data $TRAIN_DATA \\"
        echo "    --val_data $VAL_DATA \\"
        echo "    --output_dir ./checkpoints"
        echo ""
        echo "预测："
        echo "  python test_gdinn/predict_gdinn.py \\"
        echo "    --model_type SolvGNN \\"
        echo "    --batch_size 1000 \\"
        echo "    --hidden_dim 128 \\"
        echo "    --checkpoint ./checkpoints/best_model.pdparams \\"
        echo "    --test_data $TEST_DATA \\"
        echo "    --output_file ./real_predictions.csv"
    fi
    
    exit 0
else
    echo -e "${RED}✗ 部分测试失败${NC}"
    exit 1
fi
