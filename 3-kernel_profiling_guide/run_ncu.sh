#!/bin/bash

# 1. 检查有没有输入参数
if [ -z "$1" ]; then
    echo "❌ 错误: 请输入可执行文件的名字！"
    echo "💡 基础用法: ./run_ncu.sh <可执行文件名>"
    echo "💡 进阶用法: ./run_ncu.sh <可执行文件名> [目标内核名] [其他参数...]"
    echo "   举个栗子: ./run_ncu.sh my_app add2"
    exit 1
fi

# 2. 定义核心环境变量
EXE_NAME=$1
KERNEL_NAME=$2 # 第二个参数作为要抓取的内核名（可选）
NCU_PATH="/home/yechangxin/cuda-12.4/bin/ncu"
CURRENT_USER=$(whoami)

# --- 🌟 核心升级：统一指向 build 目录 ---
BUILD_DIR="../build/3-kernel_profiling_guide"
EXE_PATH="$BUILD_DIR/$EXE_NAME"

# 如果指定了内核名，报告名字里加上内核名，方便区分
if [ -n "$KERNEL_NAME" ]; then
    REPORT_PATH="$BUILD_DIR/${EXE_NAME}_${KERNEL_NAME}_report"
else
    REPORT_PATH="$BUILD_DIR/${EXE_NAME}_report"
fi

# 3. 检查可执行文件到底存不存在
if [ ! -f "$EXE_PATH" ]; then
    echo "❌ 错误: 找不到文件 '$EXE_PATH'。"
    echo "💡 请检查是否已经成功 make 编译，或者文件名是否拼写正确。"
    exit 1
fi

echo "======================================================"
echo "🚀 开始使用 Nsight Compute 剖析算子"
echo "🎯 目标程序: $EXE_PATH"
if [ -n "$KERNEL_NAME" ]; then
    echo "🔫 狙击内核: 仅剖析包含 '$KERNEL_NAME' 的函数"
fi
echo "📁 报告输出: ${REPORT_PATH}.ncu-rep"
echo "======================================================"

# 4. 构建 NCU 核心命令
# --import-source yes: 把 C++ 源码内嵌到报告里，换电脑也能看 Source 页面！
NCU_CMD="sudo $NCU_PATH --set full --import-source yes -f -o $REPORT_PATH"

# 如果用户输入了第二个参数，加上 -k 进行正则匹配过滤
if [ -n "$KERNEL_NAME" ]; then
    NCU_CMD="$NCU_CMD -k $KERNEL_NAME"
    # shift 2 把前两个参数移走，剩下的全当做可执行文件的参数
    shift 2 
else
    # shift 1 把第一个参数移走
    shift 1
fi

# 拼接上可执行文件和它的参数 ($@ 代表剩下的所有参数)
NCU_CMD="$NCU_CMD $EXE_PATH $@"

# 执行抓包
echo "⚙️  执行命令: $NCU_CMD"
$NCU_CMD

# 5. 修改文件所有权
sudo chown $CURRENT_USER:$CURRENT_USER ${REPORT_PATH}.ncu-rep

echo "======================================================"
echo "✅ 抓包圆满完成！"
echo "📂 报告已安静地躺在: ${REPORT_PATH}.ncu-rep"
echo "======================================================"