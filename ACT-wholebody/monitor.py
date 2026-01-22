#!/usr/bin/env python3
"""
ACT-wholebody Training Monitor
==============================
Real-time monitoring of training progress, GPU and CPU usage.
"""

import subprocess
import time
import re
import os
import sys
from datetime import datetime, timedelta

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def clear_screen():
    os.system('clear')

def get_gpu_info():
    """Get GPU usage info using nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpus = []
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    gpus.append({
                        'name': parts[0],
                        'util': int(parts[1]),
                        'mem_used': int(parts[2]),
                        'mem_total': int(parts[3]),
                        'temp': int(parts[4])
                    })
            return gpus
    except:
        pass
    return []

def get_cpu_info():
    """Get CPU usage"""
    try:
        result = subprocess.run(['top', '-bn1'], capture_output=True, text=True, timeout=5)
        for line in result.stdout.split('\n'):
            if 'Cpu(s)' in line or '%Cpu' in line:
                # Parse CPU usage
                match = re.search(r'(\d+\.?\d*)\s*(?:id|idle)', line)
                if match:
                    idle = float(match.group(1))
                    return 100 - idle
    except:
        pass
    return 0

def get_memory_info():
    """Get system memory usage"""
    try:
        result = subprocess.run(['free', '-m'], capture_output=True, text=True, timeout=5)
        for line in result.stdout.split('\n'):
            if line.startswith('Mem:'):
                parts = line.split()
                total = int(parts[1])
                used = int(parts[2])
                return used, total
    except:
        pass
    return 0, 0

def parse_training_log(log_path):
    """Parse training log to get current progress"""
    info = {
        'current_step': 0,
        'total_steps': 40000,
        'loss': None,
        'lr': None,
        'phase': 'unknown',
        'use_torque': None,
        'last_update': None,
        'epoch': None,
        'update_s': None,
    }

    try:
        with open(log_path, 'r') as f:
            content = f.read()
            lines = content.split('\n')

            # Detect phase
            if 'USE_TORQUE=true' in content or 'use_torque: true' in content.lower():
                if 'Step 2/2' in content:
                    info['phase'] = 'torque-true'
                    info['use_torque'] = True
                elif 'Step 1/2' in content:
                    info['phase'] = 'torque-false'
                    info['use_torque'] = False

            if 'use_torque: false' in content.lower() and 'Step 1/2' in content:
                info['phase'] = 'torque-false'
                info['use_torque'] = False

            # Find latest training step
            for line in reversed(lines):
                # Match pattern like: step:39900 smpl:... ep:... epch:0.77 loss:0.046
                step_match = re.search(r'step[:\s]*(\d+)', line, re.IGNORECASE)
                if step_match:
                    info['current_step'] = int(step_match.group(1))

                    # Extract loss
                    loss_match = re.search(r'loss[:\s]*([\d.]+)', line, re.IGNORECASE)
                    if loss_match:
                        info['loss'] = float(loss_match.group(1))

                    # Extract learning rate
                    lr_match = re.search(r'lr[:\s]*([\d.e\-+]+)', line, re.IGNORECASE)
                    if lr_match:
                        info['lr'] = lr_match.group(1)

                    # Extract epoch
                    epoch_match = re.search(r'epch[:\s]*([\d.]+)', line, re.IGNORECASE)
                    if epoch_match:
                        info['epoch'] = float(epoch_match.group(1))

                    # Extract update time
                    updt_match = re.search(r'updt_s[:\s]*([\d.]+)', line, re.IGNORECASE)
                    if updt_match:
                        info['update_s'] = float(updt_match.group(1))

                    info['last_update'] = datetime.now()
                    break

    except FileNotFoundError:
        pass
    except Exception as e:
        pass

    return info

def progress_bar(current, total, width=40):
    """Create a progress bar string"""
    if total == 0:
        return '[' + '-' * width + ']'

    percent = current / total
    filled = int(width * percent)
    bar = '█' * filled + '░' * (width - filled)
    return f'[{bar}]'

def format_time(seconds):
    """Format seconds to human readable string"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"

def estimate_remaining_time(current_step, total_steps, update_time):
    """Estimate remaining time based on update speed"""
    if update_time is None or update_time == 0 or current_step == 0:
        return None
    remaining_steps = total_steps - current_step
    return remaining_steps * update_time

def main():
    log_path = '/home/zeno/ACT-wholebody/ACT-wholebody/train.log'
    refresh_interval = 2  # seconds

    start_time = datetime.now()

    while True:
        clear_screen()

        # Header
        print(f"{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════════╗{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}║         ACT-wholebody Training Monitor                           ║{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}╚══════════════════════════════════════════════════════════════════╝{Colors.END}")
        print()

        # Time info
        now = datetime.now()
        elapsed = now - start_time
        print(f"  {Colors.BLUE}🕐 Current Time:{Colors.END} {now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  {Colors.BLUE}⏱  Monitor Running:{Colors.END} {str(elapsed).split('.')[0]}")
        print()

        # Training Progress
        print(f"{Colors.BOLD}{Colors.YELLOW}━━━ Training Progress ━━━{Colors.END}")
        info = parse_training_log(log_path)

        # Phase indicator
        phase_display = {
            'torque-false': f'{Colors.GREEN}Phase 1/2: USE_TORQUE=false (baseline){Colors.END}',
            'torque-true': f'{Colors.GREEN}Phase 2/2: USE_TORQUE=true (with torque){Colors.END}',
            'unknown': f'{Colors.YELLOW}Initializing...{Colors.END}'
        }
        print(f"  📍 {phase_display.get(info['phase'], 'Unknown')}")

        # Progress
        total_steps = info['total_steps']
        current_step = info['current_step']
        percent = (current_step / total_steps * 100) if total_steps > 0 else 0

        print(f"  📊 Step: {Colors.BOLD}{current_step:,}{Colors.END} / {total_steps:,} ({percent:.1f}%)")
        print(f"     {progress_bar(current_step, total_steps, 50)}")

        # Loss and LR
        if info['loss'] is not None:
            loss_color = Colors.GREEN if info['loss'] < 0.1 else Colors.YELLOW if info['loss'] < 0.5 else Colors.RED
            print(f"  📉 Loss: {loss_color}{info['loss']:.4f}{Colors.END}")
        if info['lr'] is not None:
            print(f"  📈 Learning Rate: {info['lr']}")
        if info['epoch'] is not None:
            print(f"  🔄 Epoch: {info['epoch']:.2f}")

        # Time estimate
        if info['update_s'] is not None:
            print(f"  ⚡ Step Time: {info['update_s']:.3f}s/step")
            remaining = estimate_remaining_time(current_step, total_steps, info['update_s'])
            if remaining is not None:
                # For phase 1, add phase 2 time
                if info['phase'] == 'torque-false':
                    remaining += total_steps * info['update_s']  # Add phase 2
                    print(f"  ⏳ Est. Remaining (both phases): {Colors.CYAN}{format_time(remaining)}{Colors.END}")
                else:
                    print(f"  ⏳ Est. Remaining: {Colors.CYAN}{format_time(remaining)}{Colors.END}")
        print()

        # GPU Info
        print(f"{Colors.BOLD}{Colors.YELLOW}━━━ GPU Status ━━━{Colors.END}")
        gpus = get_gpu_info()
        if gpus:
            for i, gpu in enumerate(gpus):
                util_color = Colors.RED if gpu['util'] > 90 else Colors.YELLOW if gpu['util'] > 70 else Colors.GREEN
                mem_percent = gpu['mem_used'] / gpu['mem_total'] * 100
                mem_color = Colors.RED if mem_percent > 90 else Colors.YELLOW if mem_percent > 70 else Colors.GREEN
                temp_color = Colors.RED if gpu['temp'] > 80 else Colors.YELLOW if gpu['temp'] > 70 else Colors.GREEN

                print(f"  🎮 GPU {i}: {gpu['name']}")
                print(f"     Utilization: {util_color}{gpu['util']:3d}%{Colors.END} {progress_bar(gpu['util'], 100, 20)}")
                print(f"     Memory:      {mem_color}{gpu['mem_used']:,} / {gpu['mem_total']:,} MB ({mem_percent:.1f}%){Colors.END}")
                print(f"     Temperature: {temp_color}{gpu['temp']}°C{Colors.END}")
        else:
            print(f"  {Colors.RED}No GPU detected{Colors.END}")
        print()

        # CPU & Memory
        print(f"{Colors.BOLD}{Colors.YELLOW}━━━ System Status ━━━{Colors.END}")
        cpu_usage = get_cpu_info()
        cpu_color = Colors.RED if cpu_usage > 90 else Colors.YELLOW if cpu_usage > 70 else Colors.GREEN
        print(f"  💻 CPU Usage: {cpu_color}{cpu_usage:.1f}%{Colors.END} {progress_bar(cpu_usage, 100, 20)}")

        mem_used, mem_total = get_memory_info()
        if mem_total > 0:
            mem_percent = mem_used / mem_total * 100
            mem_color = Colors.RED if mem_percent > 90 else Colors.YELLOW if mem_percent > 70 else Colors.GREEN
            print(f"  🧠 RAM Usage: {mem_color}{mem_used:,} / {mem_total:,} MB ({mem_percent:.1f}%){Colors.END}")
        print()

        # Footer
        print(f"{Colors.BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.END}")
        print(f"  {Colors.CYAN}Log file:{Colors.END} {log_path}")
        print(f"  {Colors.CYAN}Press Ctrl+C to exit monitor (training continues in background){Colors.END}")

        try:
            time.sleep(refresh_interval)
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Monitor stopped. Training continues in background.{Colors.END}")
            print(f"To check training: tail -f {log_path}")
            break

if __name__ == '__main__':
    main()
