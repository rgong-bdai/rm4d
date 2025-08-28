import os
import argparse
from contextlib import redirect_stdout
import numpy as np
import torch
from tqdm import tqdm

from exp_utils import get_map_type_from_exp_dir, get_sample_points_from_exp_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str,
                        help='directory of experiment')
    parser.add_argument('eval_data_dir', type=str,
                        help='directory with evaluation data for '
                             'corresponding robot')
    parser.add_argument('--batch_size', type=int, default=10000,
                        help='batch size for GPU processing')
    parser.add_argument('--use_gpu', action='store_true',
                        help='use GPU acceleration')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device to use (cuda or cpu)')

    return parser.parse_args()


def print_confusion_matrix(gt, pred, print_to=None):
    """
    Assumes numpy arrays with binary values, prints on console.
    """
    tp = np.bitwise_and(gt == 1, pred == 1).sum() / len(gt)
    tn = np.bitwise_and(gt == 0, pred == 0).sum() / len(gt)
    fp = np.bitwise_and(gt == 0, pred == 1).sum() / len(gt)
    fn = np.bitwise_and(gt == 1, pred == 0).sum() / len(gt)

    acc = tp + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    if print_to is not None:
        with open(print_to, 'w') as f:
            with redirect_stdout(f):
                print('confusion matrix:')
                print('        | gt 1  |  gt 0 |')
                print('--------|-------|-------|-------')
                print(f' pred 1 | {tp:.3f} | {fp:.3f} | {tp + fp:.3f}')
                print(f' pred 0 | {fn:.3f} | {tn:.3f} | {fn + tn:.3f}')
                print('--------|-------|-------|-------')
                print(f'        | {tp + fn:.3f} | {fp + tn:.3f} | '
                      f'{tp + tn + fp + fn:.3f}')

                print('metrics:')
                print(f'accuracy:\t {acc:.3f}')
                print(f'precision:\t {precision:.3f}')
                print(f'recall:\t {recall:.3f}')
                print(f'FPR:\t {false_positive_rate:.3f}')

    print('confusion matrix:')
    print('        | gt 1  |  gt 0 |')
    print('--------|-------|-------|-------')
    print(f' pred 1 | {tp:.3f} | {fp:.3f} | {tp + fp:.3f}')
    print(f' pred 0 | {fn:.3f} | {tn:.3f} | {fn + tn:.3f}')
    print('--------|-------|-------|-------')
    print(f'        | {tp + fn:.3f} | {fp + tn:.3f} | '
          f'{tp + tn + fp + fn:.3f}')

    print('metrics:')
    print(f'accuracy:\t {acc:.3f}')
    print(f'precision:\t {precision:.3f}')
    print(f'recall:\t {recall:.3f}')
    print(f'FPR:\t {false_positive_rate:.3f}')

    return acc, precision, recall, false_positive_rate


def batch_reachability_check_gpu(rmap, poses, batch_size=10000, device='cuda'):
    """
    GPU-accelerated batch reachability checking.
    
    Args:
        rmap: ReachabilityMap4DGPU instance
        poses: (N, 4, 4) array of poses
        batch_size: batch size for processing
        device: device to use
        
    Returns:
        (N,) boolean array indicating which poses are reachable
    """
    n_samples = len(poses)
    reachable_by_map = np.zeros(n_samples, dtype=bool)
    
    # Process in batches
    for i in tqdm(range(0, n_samples, batch_size),
                  desc="GPU batch processing"):
        end_idx = min(i + batch_size, n_samples)
        batch_poses = poses[i:end_idx]
        
        try:
            # Use GPU-accelerated batch checking
            batch_results = rmap.are_poses_reachable(batch_poses)
            
            # Convert to numpy if it's a tensor
            if isinstance(batch_results, torch.Tensor):
                batch_results = batch_results.cpu().numpy()
            
            reachable_by_map[i:end_idx] = batch_results
            
        except Exception as e:
            print(f'Error in batch {i//batch_size}: {e}')
            # Fall back to CPU processing for this batch
            for j, pose in enumerate(batch_poses):
                try:
                    idcs = rmap.get_indices_for_ee_pose(pose)
                    reachable_by_map[i + j] = rmap.is_reachable(idcs)
                except IndexError:
                    print(f'{i + j}: pose is not covered by map.')
                    reachable_by_map[i + j] = False
    
    return reachable_by_map


def main(args):
    exp_dir = args.exp_dir
    eval_data_dir = args.eval_data_dir
    batch_size = args.batch_size
    use_gpu = args.use_gpu
    device = args.device

    # Check GPU availability
    if use_gpu and not torch.cuda.is_available():
        print("Warning: GPU requested but CUDA not available. Using CPU.")
        use_gpu = False
        device = 'cpu'

    # Get eval data
    tfs_ee = np.load(os.path.join(eval_data_dir, 'poses.npy'))
    reachable_by_ik = np.load(os.path.join(eval_data_dir,
                                          'reachable_by_ik.npy'))
    n_samples = len(reachable_by_ik)

    print(f"Processing {n_samples} evaluation poses")
    print(f"Using {'GPU' if use_gpu else 'CPU'} acceleration")
    print(f"Batch size: {batch_size}")

    # Type of map - use GPU version if available
    map_type = get_map_type_from_exp_dir(exp_dir)
    
    # Try to import GPU version
    try:
        from rm4d import ReachabilityMap4DGPU
        if use_gpu:
            # Use GPU version if requested and available
            gpu_map_type = ReachabilityMap4DGPU
        else:
            gpu_map_type = map_type
    except ImportError:
        print("Warning: GPU version not available, using CPU version")
        gpu_map_type = map_type

    # Get individual sample points from experiment
    sample_points = get_sample_points_from_exp_dir(exp_dir)
    n_points = len(sample_points)

    accuracy = np.empty(n_points, dtype=float)
    precision = np.empty(n_points, dtype=float)
    recall = np.empty(n_points, dtype=float)
    false_positive_rate = np.empty(n_points, dtype=float)

    for i in range(n_points):
        cur_dir = os.path.join(exp_dir, f'{sample_points[i]}')
        print(f"\nProcessing sample point {i+1}/{n_points}: "
              f"{sample_points[i]}")

        # Load the map
        if use_gpu and gpu_map_type != map_type:
            rmap = gpu_map_type.from_file(os.path.join(cur_dir, 'rmap.npy'),
                                         use_gpu=True)
        else:
            rmap = map_type.from_file(os.path.join(cur_dir, 'rmap.npy'))

        # Use GPU-accelerated batch processing if available
        if use_gpu and hasattr(rmap, 'are_poses_reachable'):
            reachable_by_map = batch_reachability_check_gpu(
                rmap, tfs_ee, batch_size=batch_size, device=device)
        else:
            # Fall back to CPU processing
            reachable_by_map = np.zeros(n_samples, dtype=bool)
            for j in tqdm(range(n_samples), desc="CPU processing"):
                try:
                    idcs = rmap.get_indices_for_ee_pose(tfs_ee[j])
                    reachable_by_map[j] = rmap.is_reachable(idcs)
                except IndexError:
                    print(f'{j}: pose is not covered by map.')
                    reachable_by_map[j] = False

        # Calculate metrics
        fn = os.path.join(cur_dir, 'confusion_matrix.txt')
        acc, prec, rec, fpr = print_confusion_matrix(
            reachable_by_ik, reachable_by_map, print_to=fn)
        accuracy[i] = acc
        precision[i] = prec
        recall[i] = rec
        false_positive_rate[i] = fpr

    # Save results
    results = np.array([sample_points, accuracy, precision, recall,
                       false_positive_rate])
    results_fn = os.path.join(exp_dir, 'accuracy_metrics.npy')
    np.save(results_fn, results)
    
    print(f"\nResults saved to {results_fn}")
    print("Final metrics:")
    print(f"Average accuracy: {np.mean(accuracy):.3f}")
    print(f"Average precision: {np.mean(precision):.3f}")
    print(f"Average recall: {np.mean(recall):.3f}")
    print(f"Average FPR: {np.mean(false_positive_rate):.3f}")


if __name__ == '__main__':
    main(parse_args()) 