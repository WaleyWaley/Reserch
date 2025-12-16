import os
import sys

# 确保 Keras 后端设置在最前面
os.environ['KERAS_BACKEND'] = 'torch'

print("="*80)
print("开始进行环境诊断...")
print(f"当前 Python 解释器路径: {sys.executable}")
print(f"Keras 后端设置为: {os.environ.get('KERAS_BACKEND')}")
print("-"*80)

try:
    # 【新增】首先检查最基础的 packaging 库
    print("正在检查 'packaging' 库...")
    from packaging.version import parse as parse_version
    print("✅ 'packaging' 库已安装。")

    # 然后再继续检查其他库
    print("\n正在尝试导入 Keras...")
    import keras
    print(f"✅ Keras 导入成功，版本: {keras.__version__}")
    
    if parse_version(keras.__version__) < parse_version("3.0.0"):
        print("❌ 警告：Keras 版本过低，需要 V3.0.0 或更高版本。")
    else:
        print("✅ Keras 版本符合要求。")

    print("\n正在尝试从 Keras 导入 'ops'...")
    from keras import ops
    print("✅ 'keras.ops' 导入成功！")

    print("\n正在尝试导入 TensorFlow...")
    import tensorflow
    print(f"✅ TensorFlow 导入成功，版本: {tensorflow.__version__}")

    print("\n正在尝试导入 PyTorch...")
    import torch
    print(f"✅ PyTorch 导入成功，版本: {torch.__version__}")

    print("\n正在尝试导入 tkan...")
    from tkan import TKAN
    print("✅ tkan 导入成功！")
    
    print("\n" + "="*80)
    print("🎉 诊断完成：所有核心库均可正常导入，环境配置看起来是正确的！")
    print("="*80)

except ImportError as e:
    # 捕获导入错误，并提供更具体的指导
    error_message = str(e)
    print("\n" + "="*80)
    print("❌ 诊断失败：在导入过程中发生错误。")
    print(f"错误类型: {type(e).__name__}")
    print(f"错误信息: {error_message}")
    
    if "No module named 'packaging'" in error_message:
        print("\n原因分析：缺少 'packaging' 库，这是一个检查软件版本号所必需的基础工具。")
        print("\n解决方案：请在您的 conda 环境中运行以下命令来安装：")
        print("pip install packaging")

    # ... (保留之前对 tensorflow, torch, keras 版本的检查和提示) ...
        
    print("="*80)
except Exception as e:
    print("\n" + "="*80)
    print("❌ 诊断失败：发生未知错误。")
    print(f"错误类型: {type(e).__name__}")
    print(f"错误信息: {e}")
    print("="*80)

