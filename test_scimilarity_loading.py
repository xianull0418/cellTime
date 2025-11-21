
import torch
from omegaconf import OmegaConf
from models.ae import AESystem
import os

def test_loading():
    # 模拟配置
    config_path = "config/ae_scimilarity.yaml"
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        return

    cfg = OmegaConf.load(config_path)
    
    # 模拟数据路径（虽然这里不会用到）
    cfg.data.data_path = "dummy.h5ad"
    
    # 设置为 CPU 运行
    cfg.accelerator.accelerator = "cpu"
    
    print("=== 初始化 AESystem ===")
    try:
        system = AESystem(cfg)
        print("\n=== 初始化成功 ===")
        
        # 检查权重是否加载（通过打印第一层和中间层的部分权重来验证，这里主要看日志输出）
        # 日志应该显示 "成功加载 ... 个参数张量" 和 "跳过了 ... 个不匹配/重置的层"
        
    except Exception as e:
        print(f"\n=== 初始化失败 ===\n{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_loading()

