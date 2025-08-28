import os
import argparse
import pathlib
import numpy as np

from rm4d import HitStats
from rm4d.robots import Simulator

from exp_utils import robot_types, map_types, constructor_types, Timer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--robot_type', choices=list(robot_types.keys()), help='robot type.', default='franka')
    parser.add_argument('-m', '--map_type', choices=list(map_types.keys()), help='map type.', default='rm4d')
    parser.add_argument('-c', '--constructor_type', choices=list(constructor_types.keys()), help='constructor type',
                        default='joint')
    parser.add_argument('-n', '--num_samples', type=int, help='maximum number of samples', default=int(2e7))
    parser.add_argument('-e', '--every', type=int, help='save map and stats every n samples', default=int(2e6))
    parser.add_argument('-s', '--seed', type=int, default=42)
    parser.add_argument('-v', '--voxel_res', type=float, default=0.05)

    return parser.parse_args()


def main(args):
    voxel_res = args.voxel_res
    seed = args.seed
    n_section_samples = args.every
    n_save_points = args.num_samples // args.every
    robot_type = args.robot_type
    map_type = args.map_type
    constructor_type = args.constructor_type
    exp_dir = os.path.join('data', f'{map_type}_{robot_type}_{constructor_type}_{seed}_{voxel_res}')
    pathlib.Path(exp_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(exp_dir, 'experiment.log')

    def log(str_arg, mode='a'):
        with open(log_file, mode) as f:
            f.write(str_arg + '\n')
        print(str_arg)

    log('configuration', mode='w')
    for arg, val in sorted(vars(args).items()):
        log(f'{arg}: {val}')
    log(f'{n_save_points} save points.')
    log('setting up...')

    # set up everything
    sim = Simulator(with_gui=False)
    robot = robot_types[robot_type](sim)
    z_limits = [0, robot.range_z]
    xy_limits = [-robot.range_radius, robot.range_radius]

    rmap = map_types[map_type](xy_limits=xy_limits, z_limits=z_limits, voxel_res=voxel_res)
    hit_stats = HitStats(rmap.shape, record_every=n_section_samples)
    constructor = constructor_types[constructor_type](rmap, robot, seed=seed)
    t = Timer()

    log('*********************************')
    for i in range(n_save_points):
        t.start('sample')
        constructor.sample(n_samples=n_section_samples, prevent_collisions=True, hit_stats=hit_stats)
        t.stop('sample')

        # save results
        n_samples_so_far = (i+1) * n_section_samples
        save_dir = os.path.join(exp_dir, f'{n_samples_so_far}')
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

        hit_stats.to_file(save_dir)
        rmap.to_file(os.path.join(save_dir, f'rmap'))
        log(f'{i+1}/{n_save_points}: {n_samples_so_far} samples after {t.timers["sample"]:.2f} seconds.')
        log(f'\t{n_samples_so_far/t.timers["sample"]:.4f} samples per second')

    hit_stats.to_file(exp_dir)
    sim.disconnect()


if __name__ == '__main__':
    main(parse_args())
