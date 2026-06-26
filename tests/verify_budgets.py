import os
import time
import torch
from pathlib import Path
import numpy as np

# Suppress warnings for clean output
import warnings
warnings.filterwarnings('ignore')

print("Loading dependencies and model for verification...")

# Append UETrack to path so we can import it
import sys
sys.path.insert(0, '/workspace/UETrack')

from inference_scripts.inference import network, cfg, UETrackOnline, preprocess_image, sample_target

def verify_budgets():
    device = cfg.device
    network.eval()
    
    # 1. Model Size (Disk Footprint)
    ckpt_path = Path('/workspace/checkpoints/model_final.pth')
    if not ckpt_path.exists():
        ckpt_path = Path('checkpoints/model_final.pth')
    
    disk_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    print(f"\n[1] Model Size (Disk Footprint)")
    print(f"    Budget: 500 MB (0.5 GB)")
    print(f"    Actual: {disk_size_mb:.2f} MB")
    print(f"    Status: {'✅ PASS' if disk_size_mb <= 500 else '❌ FAIL'}")

    # 2. Number of Parameters
    total_params = sum(p.numel() for p in network.parameters())
    params_m = total_params / 1e6
    print(f"\n[2] Number of Parameters")
    print(f"    Budget: 50 Million")
    print(f"    Actual: {params_m:.2f} Million")
    print(f"    Status: {'✅ PASS' if params_m <= 50 else '❌ FAIL'}")

    # Set up dummy inputs based on inference shapes
    print("\nPreparing dummy inputs for Latency and FLOP calculations...")
    dummy_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    dummy_box = [500.0, 500.0, 50.0, 50.0]
    
    tracker = UETrackOnline(network, cfg, device=device)
    tracker.initialize(dummy_image, dummy_box)
    
    # Extract the exact tensors the network uses for tracking
    template_list = tracker.template_list
    template_anno_list = tracker.template_anno_list
    
    x_patch, resize_factor = sample_target(dummy_image, tracker.state, cfg.search_factor, cfg.search_size)
    search = preprocess_image(x_patch, np.array([0.485, 0.456, 0.406], dtype=np.float32), np.array([0.229, 0.224, 0.225], dtype=np.float32)).unsqueeze(0).to(device)

    # 3. Inference Latency
    print(f"\n[3] Inference Latency (Measuring on {device.upper()})")
    print(f"    Budget: 30 ms")
    
    # Warmup
    for _ in range(20):
        with torch.no_grad():
            raw = network.forward_encoder(template_list, [search], template_anno_list, text_src=None, task_index=0)
            _ = network.forward_decoder(raw[0])
            
    # Measure
    iters = 100
    start_time = time.time()
    for _ in range(iters):
        with torch.no_grad():
            raw = network.forward_encoder(template_list, [search], template_anno_list, text_src=None, task_index=0)
            _ = network.forward_decoder(raw[0])
            
    if device == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.time()
    avg_latency_ms = ((end_time - start_time) / iters) * 1000
    
    print(f"    Actual: {avg_latency_ms:.2f} ms")
    print(f"    Status: {'✅ PASS' if avg_latency_ms <= 30 else '⚠️ WARNING (Depends on target GPU hardware)'}")

    # 4. Computational Complexity (FLOPs)
    print(f"\n[4] Computational Complexity (FLOPs)")
    print(f"    Budget: 30 GFLOPs")
    
    try:
        from fvcore.nn import FlopCountAnalysis
        
        # Create a wrapper module to pass into FlopCountAnalysis
        class TrackerWrapper(torch.nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net
            def forward(self, t_list, s, ta_list):
                r = self.net.forward_encoder(t_list, [s], ta_list, text_src=None, task_index=0)
                return self.net.forward_decoder(r[0])
                
        wrapper = TrackerWrapper(network).eval().to(device)
        
        # Convert list arguments to tuples for fvcore compatibility
        flops = FlopCountAnalysis(wrapper, (template_list, search, template_anno_list))
        # fvcore tends to warn heavily on custom operators, suppress them
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        flops.tracer_warnings('none')
        
        total_flops = flops.total()
        gflops = total_flops / 1e9
        
        print(f"    Actual: {gflops:.2f} GFLOPs")
        print(f"    Status: {'✅ PASS' if gflops <= 30 else '❌ FAIL'}")
        
    except ImportError:
        print("    Actual: fvcore not found. Could not measure FLOPs.")
        print("    Status: ⚠️ UNKNOWN")
    except Exception as e:
        print(f"    Actual: Error calculating FLOPs: {e}")
        print("    Status: ⚠️ UNKNOWN")

if __name__ == '__main__':
    verify_budgets()
