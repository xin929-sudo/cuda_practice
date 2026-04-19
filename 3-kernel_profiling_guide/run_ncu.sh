#!/bin/bash

# 1. 检查有没有输入参数
if [ -z "$1" ]; then
    echo "❌ 错误: 请输入可执行文件的名字！"
    echo "💡 用法: ./run_ncu.sh <可执行文件名>"
    exit 1
fi

# 2. 定义核心环境变量
EXE_NAME=$1
NCU_PATH="/home/yechangxin/cuda-12.4/bin/ncu"
CURRENT_USER=$(whoami)

# --- 🌟 核心升级：统一指向 build 目录 ---
# 把目标文件夹路径单独拎出来，方便以后修改
BUILD_DIR="../build/3-kernel_profiling_guide"

EXE_PATH="$BUILD_DIR/$EXE_NAME"
# 把报告的输出路径也强行指派到 build 目录里
REPORT_PATH="$BUILD_DIR/${EXE_NAME}_report"

# 3. 检查可执行文件到底存不存在
if [ ! -f "$EXE_PATH" ]; then
    echo "❌ 错误: 找不到文件 '$EXE_PATH'。"
    echo "💡 请检查是否已经成功 make 编译，或者文件名是否拼写正确。"
    exit 1
fi

echo "======================================================"
echo "🚀 开始使用 Nsight Compute 剖析算子: $EXE_NAME"
echo "🎯 目标程序: $EXE_PATH"
echo "📁 报告输出: $BUILD_DIR/"
echo "======================================================"

# 4. 执行抓包 (注意 -o 后面换成了带有完整 build 路径的 $REPORT_PATH)
sudo $NCU_PATH --set full -f -o $REPORT_PATH $EXE_PATH

# 5. 修改文件所有权 (注意 chown 找的也是 build 目录下的文件)
sudo chown $CURRENT_USER:$CURRENT_USER ${REPORT_PATH}.ncu-rep

echo "======================================================"
echo "✅ 抓包圆满完成！"
echo "📂 报告已安静地躺在: ${REPORT_PATH}.ncu-rep"
echo "======================================================"