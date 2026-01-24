#!/usr/bin/env python3
"""
ACT-fullbody Loss Comparison Plot

This script reads training logs from all 4 configurations and generates
a comparison plot showing loss curves over training steps.
"""

import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_training_log(log_file: str) -> tuple[list, list]:
    """Parse training log file to extract steps and losses.

    Supports multiple log formats:
    1. JSON format: {"step": N, "loss": X, ...}
    2. Text format: "step: N, loss: X" or similar patterns
    """
    steps = []
    losses = []

    if not os.path.exists(log_file):
        print(f"Warning: Log file not found: {log_file}")
        return steps, losses

    with open(log_file, 'r') as f:
        content = f.read()

    # Try JSON format first (one JSON object per line)
    try:
        for line in content.strip().split('\n'):
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                data = json.loads(line)
                if 'step' in data and 'loss' in data:
                    steps.append(data['step'])
                    losses.append(data['loss'])
        if steps:
            return steps, losses
    except json.JSONDecodeError:
        pass

    # Try parsing from text log (common patterns)
    # Pattern 1: "step: 100, loss: 0.123"
    pattern1 = r'step[:\s]+(\d+).*?loss[:\s]+([\d.]+)'
    matches = re.findall(pattern1, content, re.IGNORECASE)
    if matches:
        for step, loss in matches:
            steps.append(int(step))
            losses.append(float(loss))
        return steps, losses

    # Pattern 2: "Step 100 | Loss: 0.123"
    pattern2 = r'Step\s+(\d+).*?Loss[:\s]+([\d.]+)'
    matches = re.findall(pattern2, content)
    if matches:
        for step, loss in matches:
            steps.append(int(step))
            losses.append(float(loss))
        return steps, losses

    # Pattern 3: Look for l1_loss in JSON-like structures
    pattern3 = r'"step":\s*(\d+).*?"l1_loss":\s*([\d.]+)'
    matches = re.findall(pattern3, content)
    if matches:
        for step, loss in matches:
            steps.append(int(step))
            losses.append(float(loss))
        return steps, losses

    print(f"Warning: Could not parse log format in {log_file}")
    return steps, losses


def find_log_file(checkpoint_dir: str) -> str | None:
    """Find the training log file in a checkpoint directory."""
    checkpoint_path = Path(checkpoint_dir)

    # Common log file names
    candidates = [
        'training_log.json',
        'training_log.txt',
        'train.log',
        'log.txt',
        'metrics.json',
    ]

    for candidate in candidates:
        log_path = checkpoint_path / candidate
        if log_path.exists():
            return str(log_path)

    # Look for any .json or .log file
    for ext in ['*.json', '*.log', '*.txt']:
        files = list(checkpoint_path.glob(ext))
        if files:
            return str(files[0])

    return None


def main():
    script_dir = Path(__file__).parent
    checkpoints_dir = script_dir / 'checkpoints'

    configs = [
        ('baseline', 'blue', 'Baseline (no base, no torque)'),
        ('torque-only', 'orange', 'Torque Only'),
        ('base-only', 'green', 'Base Only'),
        ('fullbody', 'red', 'Fullbody (base + torque)'),
    ]

    plt.figure(figsize=(12, 6))

    has_data = False
    for config_name, color, label in configs:
        checkpoint_dir = checkpoints_dir / f'ACT-fullbody-{config_name}'

        # Try to find log file
        log_file = find_log_file(str(checkpoint_dir))

        if log_file is None:
            # Also try the main train.log and filter by config name
            main_log = script_dir / 'train.log'
            if main_log.exists():
                log_file = str(main_log)
            else:
                print(f"No log file found for {config_name}")
                continue

        steps, losses = parse_training_log(log_file)

        if steps and losses:
            plt.plot(steps, losses, label=label, color=color, alpha=0.8, linewidth=1.5)
            has_data = True
            print(f"Loaded {len(steps)} data points for {config_name}")
        else:
            print(f"No data found for {config_name}")

    if has_data:
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('ACT-fullbody: Loss Comparison (4 Configurations)', fontsize=14)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = script_dir / 'loss_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot to: {output_path}")

        # Also save as PDF for higher quality
        pdf_path = script_dir / 'loss_comparison.pdf'
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved PDF to: {pdf_path}")
    else:
        print("\nNo training data found. Please run training first.")
        print("Expected checkpoint directories:")
        for config_name, _, _ in configs:
            print(f"  - {checkpoints_dir}/ACT-fullbody-{config_name}/")


if __name__ == '__main__':
    main()
