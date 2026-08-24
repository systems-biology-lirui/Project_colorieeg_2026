import tensorflow as tf

# 1. 检查版本
print("TF Version:", tf.__version__)

# 2. 检查 GPU 是否就绪
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"🎉 成功！检测到 {len(gpus)} 个 GPU: {gpus}")
else:
    print("❌ 失败：未检测到 GPU，请检查步骤 3 和 4")