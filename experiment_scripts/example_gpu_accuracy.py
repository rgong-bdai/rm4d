#!/usr/bin/env python3
"""
Example script demonstrating GPU-accelerated accuracy calculation for reachability maps.

This script shows how to use the calculate_accuracy_gpu.py script to evaluate
the accuracy of GPU-based reachability maps.
"""

import os
import sys
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Example GPU-accelerated accuracy calculation")
    parser.add_argument('--exp_dir', type=str, required=True,
                       help='Directory containing experiment data')
    parser.add_argument('--eval_data_dir', type=str, required=True,
                       help='Directory containing evaluation data')
    parser.add_argument('--batch_size', type=int, default=10000,
                       help='Batch size for GPU processing')
    parser.add_argument('--use_gpu', action='store_true',
                       help='Enable GPU acceleration')
    
    args = parser.parse_args()
    
    # Check if directories exist
    if not os.path.exists(args.exp_dir):
        print(f"Error: Experiment directory {args.exp_dir} does not exist")
        sys.exit(1)
    
    if not os.path.exists(args.eval_data_dir):
        print(f"Error: Evaluation data directory {args.eval_data_dir} "
              f"does not exist")
        sys.exit(1)
    
    # Build command
    cmd = [
        'python', 'calculate_accuracy_gpu.py',
        args.exp_dir,
        args.eval_data_dir,
        '--batch_size', str(args.batch_size)
    ]
    
    if args.use_gpu:
        cmd.append('--use_gpu')
    
    print("Running GPU-accelerated accuracy calculation...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run the script
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\nAccuracy calculation completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error running accuracy calculation: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 