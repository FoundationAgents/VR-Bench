"""
3D Maze 颜色处理器
负责从皮肤目录加载颜色配置
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional


# 必需的颜色键
REQUIRED_COLOR_KEYS = ['start_pos', 'goal_pos', 'default_cube', 'ball', 'ball_edge']


def load_colors_from_skin(skin_folder: str) -> Dict[str, str]:
    """
    从皮肤目录加载颜色配置

    Args:
        skin_folder: 皮肤目录路径

    Returns:
        颜色配置字典

    Raises:
        FileNotFoundError: 皮肤目录或 colors.json 不存在
        ValueError: colors.json 格式错误或缺少必需的颜色键
    """
    if not skin_folder:
        raise FileNotFoundError("No skin folder specified")

    skin_path = Path(skin_folder)
    if not skin_path.exists():
        raise FileNotFoundError(f"Skin folder not found: {skin_folder}")

    colors_path = skin_path / 'colors.json'

    if not colors_path.exists():
        raise FileNotFoundError(f"colors.json not found in {skin_folder}")

    try:
        with open(colors_path, 'r', encoding='utf-8') as f:
            colors = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse colors.json in {skin_folder}: {e}")

    missing_keys = [k for k in REQUIRED_COLOR_KEYS if k not in colors]
    if missing_keys:
        raise ValueError(f"Missing required color keys {missing_keys} in {colors_path}")

    logging.debug(f"Loaded colors from {colors_path}")
    return colors


def load_skin_description(skin_folder: str) -> Optional[Dict[str, str]]:
    """
    从皮肤目录加载视觉描述
    
    Args:
        skin_folder: 皮肤目录路径
        
    Returns:
        视觉描述字典，如果加载失败则返回 None
    """
    if not skin_folder:
        return None
    
    desc_path = Path(skin_folder) / 'description.json'
    
    if not desc_path.exists():
        return None
    
    try:
        with open(desc_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data.get('visual_description')
        
    except Exception as e:
        logging.error(f"Failed to load description from {skin_folder}: {e}")
        return None


