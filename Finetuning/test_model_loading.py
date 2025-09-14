import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import numpy as np

def test_model_loading():
    print("=" * 50)
    print("Testing DAV2 Model Loading")
    print("=" * 50)
    
    model_id = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. モデルとプロセッサの読み込み
    print(f"\n1. Loading model: {model_id}")
    print(f"   Device: {device}")
    
    try:
        processor = AutoImageProcessor.from_pretrained(model_id)
        print("✓ Processor loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load processor: {e}")
        return False
    
    try:
        model = AutoModelForDepthEstimation.from_pretrained(model_id)
        model = model.to(device)
        print(f"✓ Model loaded successfully")
        print(f"  Model type: {type(model).__name__}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # 2. ダミー画像でのテスト推論
    print("\n2. Testing inference with dummy image...")
    try:
        # ダミー画像を作成
        dummy_img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        
        # 前処理
        inputs = processor(images=dummy_img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        print(f"✓ Input processed: {inputs['pixel_values'].shape}")
        
        # 推論
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 出力の確認
        pred_depth = outputs.predicted_depth
        print(f"✓ Output shape: {pred_depth.shape}")
        print(f"  Output range: [{pred_depth.min().item():.4f}, {pred_depth.max().item():.4f}]")
        
        # 次元の確認
        if pred_depth.dim() == 3:
            print("  Note: Output is 3D [B, H, W], will need unsqueeze for loss")
        else:
            print(f"  Output dimensions: {pred_depth.dim()}D")
            
    except Exception as e:
        print(f"✗ Failed inference test: {e}")
        return False
    
    # 3. メモリ使用量の確認
    if torch.cuda.is_available():
        print(f"\n3. GPU Memory Usage:")
        print(f"   Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        print(f"   Reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
    return True

if __name__ == "__main__":
    import sys
    success = test_model_loading()
    sys.exit(0 if success else 1)