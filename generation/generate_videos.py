#!/usr/bin/env python3
"""
统一的视频生成工具
使用GameAdapter统一接口生成所有游戏的视频
"""
import sys
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count, current_process
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.schema import UnifiedState
from core.game_adapter import GameAdapter

logging.basicConfig(level=logging.INFO, format='%(message)s')


def get_game_adapter(game_type: str) -> GameAdapter:
    """根据游戏类型获取对应的适配器"""
    if game_type == "sokoban":
        from games.sokoban.adapter import SokobanAdapter
        return SokobanAdapter()
    elif game_type == "maze":
        from games.maze.adapter import MazeAdapter
        return MazeAdapter()
    elif game_type == "pathfinder":
        from games.pathfinder.adapter import PathFinderAdapter
        return PathFinderAdapter()
    elif game_type == "trapfield":
        from games.trapfield.adapter import TrapFieldAdapter
        return TrapFieldAdapter()
    elif game_type == "maze3d":
        from games.maze3d.adapter import Maze3DAdapter
        return Maze3DAdapter()
    else:
        raise ValueError(f"Unknown game type: {game_type}")


def extract_first_frame(video_path: str, image_path: str) -> bool:
    """从视频提取第一帧"""
    try:
        import imageio.v2 as imageio
        reader = imageio.get_reader(video_path)
        first_frame = reader.get_data(0)
        reader.close()
        imageio.imwrite(image_path, first_frame)
        return True
    except Exception as e:
        logging.warning(f"Failed to extract first frame: {e}")
        return False


def process_single_path(args):
    """处理单条路径（生成一个视频）"""
    state_path, game_type, path_id, video_path, skip_existing, assets_folder = args

    try:
        if skip_existing and Path(video_path).exists():
            return (path_id, 'skipped', None)

        # 使用adapter的统一接口
        adapter = get_game_adapter(game_type)
        success = adapter.generate_video(
            str(state_path),
            str(video_path),
            assets_folder=assets_folder
        )

        if success and Path(video_path).exists():
            return (path_id, 'success', None)
        else:
            return (path_id, 'failed', '视频生成失败')

    except Exception as e:
        return (path_id, 'failed', str(e))


def process_single_state(
    state_path: Path,
    game_type: str,
    skip_existing: bool = False,
    verbose: bool = True,
    num_workers: int = 4,
    assets_folder: Optional[str] = None
) -> dict:
    """处理单个state文件，生成所有最优路径的视频"""
    from generation.path_finder import find_optimal_paths
    
    stats = {'total_paths': 0, 'success': 0, 'failed': 0, 'skipped': 0, 'errors': []}

    try:
        state = UnifiedState.load(str(state_path))
        paths = find_optimal_paths(state, game_type)

        if game_type == 'pathfinder':
            paths = [p[0] for p in paths]

        stats['total_paths'] = len(paths)

        if not paths:
            if verbose:
                logging.warning(f"⚠️  {state_path.name}: 未找到路径")
            return stats

        state_dir = state_path.parent
        videos_dir = state_dir.parent / 'videos'
        images_dir = state_dir.parent / 'images'
        videos_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        base_name = state_path.stem
        
        # 为每条路径生成任务（仅第一条路径，因为adapter会自动查找最优路径）
        # 对于多条路径的情况，可以扩展adapter接口支持指定路径
        tasks = [(state_path, game_type, 0, str(videos_dir / f"{base_name}_0.mp4"), skip_existing, assets_folder)]

        # daemon进程（外层Pool的worker）不允许再创建子进程，嵌套Pool会抛
        # AssertionError；单任务时开Pool也只是纯开销，直接串行执行
        if len(tasks) > 1 and not current_process().daemon:
            with Pool(processes=min(num_workers, len(tasks))) as pool:
                results = pool.map(process_single_path, tasks)
        else:
            results = [process_single_path(task) for task in tasks]

        first_video = videos_dir / f"{base_name}_0.mp4"
        if first_video.exists():
            extract_first_frame(str(first_video), str(images_dir / f"{base_name}.png"))

        for path_id, status, error_msg in results:
            if status == 'success':
                stats['success'] += 1
            elif status == 'skipped':
                stats['skipped'] += 1
            else:
                stats['failed'] += 1
                stats['errors'].append(f"{base_name}_{path_id}: {error_msg}")
                if verbose:
                    logging.error(f"  ❌ {base_name}_{path_id}: {error_msg}")

    except Exception as e:
        stats['errors'].append(f"{state_path.name}: {e}")
        if verbose:
            logging.error(f"❌ {state_path.name}: {e}")

    return stats


def batch_process_dataset(
    dataset_root: str,
    skip_existing: bool = False,
    verbose: bool = True,
    num_workers: int = None,
    parallel_states: int = 1,
    assets_folder: Optional[str] = None
):
    """批量处理dataset目录"""
    from functools import partial
    
    if num_workers is None:
        num_workers = max(1, cpu_count() // 2)

    dataset_path = Path(dataset_root)
    state_files = []

    for state_file in dataset_path.rglob('states/*.json'):
        path_str = str(state_file).lower()
        if 'pathfinder' in path_str or 'irregular_maze' in path_str:
            game_type = 'pathfinder'
        elif '3d' in path_str or 'maze3d' in path_str:
            game_type = 'maze3d'
        elif 'maze' in path_str:
            game_type = 'maze'
        elif 'trapfield' in path_str:
            game_type = 'trapfield'
        elif 'sokoban' in path_str:
            game_type = 'sokoban'
        else:
            continue
        state_files.append((state_file, game_type))

    logging.info(f"找到 {len(state_files)} 个state文件")
    logging.info(f"并行: {parallel_states} states, 每个{num_workers} workers")
    if assets_folder:
        logging.info(f"皮肤: {assets_folder}\n")
    else:
        logging.info(f"皮肤: 默认\n")

    by_game = {}
    for state_file, game_type in state_files:
        by_game.setdefault(game_type, []).append(state_file)

    total_stats = {'total_states': 0, 'total_paths': 0, 'success': 0, 'failed': 0, 'skipped': 0}

    for game_type in sorted(by_game.keys()):
        state_paths = by_game[game_type]
        logging.info(f"\n{'='*60}")
        logging.info(f"{game_type.upper()}: {len(state_paths)} files")
        logging.info(f"{'='*60}\n")

        sorted_paths = sorted(state_paths)

        if parallel_states > 1:
            process_func = partial(
                process_single_state,
                game_type=game_type,
                skip_existing=skip_existing,
                verbose=False,
                num_workers=num_workers,
                assets_folder=assets_folder
            )
            with Pool(processes=parallel_states) as pool:
                results = pool.map(process_func, sorted_paths)

            for state_path, stats in zip(sorted_paths, results):
                total_stats['total_states'] += 1
                for k in ['total_paths', 'success', 'failed', 'skipped']:
                    total_stats[k] += stats[k]
                if verbose:
                    # worker里verbose=False，错误信息通过stats['errors']带回来
                    marker = '✅' if stats['success'] and not stats['failed'] else '⚠️'
                    logging.info(f"{marker} {state_path.name}: {stats['total_paths']}路径 "
                               f"{stats['success']}成功 {stats['skipped']}跳过 {stats['failed']}失败")
                for err in stats['errors']:
                    logging.error(f"  ❌ {err}")
        else:
            for state_path in sorted_paths:
                if verbose:
                    logging.info(f"📄 {state_path.relative_to(dataset_path)}")
                stats = process_single_state(state_path, game_type, skip_existing, verbose, num_workers, assets_folder)
                total_stats['total_states'] += 1
                for k in ['total_paths', 'success', 'failed', 'skipped']:
                    total_stats[k] += stats[k]

    logging.info(f"\n{'='*60}")
    logging.info("完成")
    logging.info(f"{'='*60}")
    logging.info(f"States: {total_stats['total_states']}, Paths: {total_stats['total_paths']}")
    logging.info(f"成功: {total_stats['success']}, 跳过: {total_stats['skipped']}, 失败: {total_stats['failed']}")
    logging.info(f"{'='*60}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='批量生成GT视频')
    parser.add_argument('dataset', help='Dataset根目录')
    parser.add_argument('--skip-existing', action='store_true', help='跳过已存在文件')
    parser.add_argument('--quiet', action='store_true', help='静默模式')
    parser.add_argument('--workers', type=int, default=None,
                       help=f'每个state的workers (默认: {max(1, cpu_count()//2)})')
    parser.add_argument('--parallel-states', type=int, default=1, help='并行处理的states数')
    parser.add_argument('--skin', type=str, default=None,
                       help='皮肤文件夹路径 (例如: skins/maze/5)')
    args = parser.parse_args()

    batch_process_dataset(
        args.dataset,
        args.skip_existing,
        not args.quiet,
        args.workers,
        args.parallel_states,
        args.skin
    )


if __name__ == '__main__':
    main()
