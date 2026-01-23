#!/usr/bin/env python3
"""
Test script to compare pyav vs torchcodec video decoding performance.
This script does NOT interfere with running training.
"""

import time
import sys
from pathlib import Path

# Add lerobot to path
sys.path.insert(0, '/workspace/ACT-wholebody-torque/lerobot/src')

from lerobot.datasets.video_utils import decode_video_frames
import torch

def test_video_backend(video_path, timestamps, backend_name, tolerance_s=0.05):
    """Test a single backend and return timing + results."""
    print(f"\n{'='*60}")
    print(f"Testing backend: {backend_name}")
    print(f"{'='*60}")

    # Warm up (first call might be slower due to initialization)
    try:
        _ = decode_video_frames(video_path, timestamps[:1], tolerance_s, backend=backend_name)
        print("✓ Warm-up successful")
    except Exception as e:
        print(f"✗ Warm-up failed: {e}")
        return None, None

    # Actual test with timing
    start_time = time.time()
    try:
        frames = decode_video_frames(video_path, timestamps, tolerance_s, backend=backend_name)
        elapsed = time.time() - start_time

        print(f"✓ Decoded {len(frames)} frames")
        print(f"  Frame shape: {frames.shape}")
        print(f"  Frame dtype: {frames.dtype}")
        print(f"  Frame range: [{frames.min():.3f}, {frames.max():.3f}]")
        print(f"  Time elapsed: {elapsed:.4f}s")
        print(f"  Time per frame: {elapsed/len(timestamps):.4f}s")

        return frames, elapsed
    except Exception as e:
        print(f"✗ Decoding failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def compare_frames(frames1, frames2, name1, name2):
    """Compare two sets of frames."""
    print(f"\n{'='*60}")
    print(f"Comparing {name1} vs {name2}")
    print(f"{'='*60}")

    if frames1 is None or frames2 is None:
        print("✗ Cannot compare - one or both backends failed")
        return

    # Check shapes
    if frames1.shape != frames2.shape:
        print(f"✗ Shape mismatch: {frames1.shape} vs {frames2.shape}")
        return

    # Compute differences
    abs_diff = torch.abs(frames1 - frames2)
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    print(f"✓ Shapes match: {frames1.shape}")
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")

    # Check if frames are similar (allowing for small decoding differences)
    if max_diff < 0.01:  # Less than 1% difference in [0,1] range
        print(f"✓ Frames are very similar (max diff < 0.01)")
    elif max_diff < 0.05:
        print(f"⚠ Frames are similar but with noticeable differences (max diff < 0.05)")
    else:
        print(f"✗ Frames have significant differences (max diff >= 0.05)")

def main():
    # Find a sample video from the dataset
    dataset_root = Path("/workspace/ACT-wholebody-torque/ACT-wholebody/data/ACT-100-wholebody")

    # Look for a video file
    video_files = list(dataset_root.rglob("*.mp4"))

    if not video_files:
        print("✗ No video files found in dataset")
        return

    # Use the first video file
    video_path = video_files[0]
    print(f"Using video: {video_path}")
    print(f"Video size: {video_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Test with a few timestamps (simulating what training does)
    # Typical ACT uses a sequence of frames
    timestamps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # 8 frames
    tolerance_s = 0.05

    print(f"\nTest parameters:")
    print(f"  Timestamps: {timestamps}")
    print(f"  Tolerance: {tolerance_s}s")

    # Test pyav
    frames_pyav, time_pyav = test_video_backend(video_path, timestamps, "pyav", tolerance_s)

    # Test torchcodec
    frames_torchcodec, time_torchcodec = test_video_backend(video_path, timestamps, "torchcodec", tolerance_s)

    # Compare results
    compare_frames(frames_pyav, frames_torchcodec, "pyav", "torchcodec")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    if time_pyav and time_torchcodec:
        speedup = time_pyav / time_torchcodec
        print(f"pyav time:       {time_pyav:.4f}s")
        print(f"torchcodec time: {time_torchcodec:.4f}s")
        print(f"Speedup:         {speedup:.2f}x")

        if speedup > 1.5:
            print(f"\n✓ torchcodec is significantly faster ({speedup:.2f}x speedup)")
            print("  Recommendation: Switch to torchcodec for training")
        elif speedup > 1.1:
            print(f"\n✓ torchcodec is moderately faster ({speedup:.2f}x speedup)")
            print("  Recommendation: Consider switching to torchcodec")
        else:
            print(f"\n⚠ torchcodec is not significantly faster ({speedup:.2f}x)")
            print("  Recommendation: May not be worth switching")

    print("\nTest completed without affecting running training.")

if __name__ == "__main__":
    main()
