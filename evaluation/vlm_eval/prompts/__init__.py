import json
from pathlib import Path
from typing import Dict

from .maze_prompt import MAZE_SYSTEM_PROMPT_TEMPLATE, MAZE_USER_PROMPT_TEMPLATE
from .sokoban_prompt import SOKOBAN_SYSTEM_PROMPT_TEMPLATE, SOKOBAN_USER_PROMPT_TEMPLATE
from .trapfield_prompt import TRAPFIELD_SYSTEM_PROMPT_TEMPLATE, TRAPFIELD_USER_PROMPT_TEMPLATE
from .pathfinder_prompt import PATHFINDER_SYSTEM_PROMPT_TEMPLATE, PATHFINDER_USER_PROMPT_TEMPLATE
from .maze3d_prompt import MAZE3D_SYSTEM_PROMPT_TEMPLATE, MAZE3D_USER_PROMPT_TEMPLATE

# Prompt 模板映射
PROMPT_TEMPLATES = {
    'maze': {
        'system': MAZE_SYSTEM_PROMPT_TEMPLATE,
        'user': MAZE_USER_PROMPT_TEMPLATE,
    },
    'sokoban': {
        'system': SOKOBAN_SYSTEM_PROMPT_TEMPLATE,
        'user': SOKOBAN_USER_PROMPT_TEMPLATE,
    },
    'trapfield': {
        'system': TRAPFIELD_SYSTEM_PROMPT_TEMPLATE,
        'user': TRAPFIELD_USER_PROMPT_TEMPLATE,
    },
    'pathfinder': {
        'system': PATHFINDER_SYSTEM_PROMPT_TEMPLATE,
        'user': PATHFINDER_USER_PROMPT_TEMPLATE,
    },
    'maze3d': {
        'system': MAZE3D_SYSTEM_PROMPT_TEMPLATE,
        'user': MAZE3D_USER_PROMPT_TEMPLATE,
    },
}

# 游戏名称别名
GAME_ALIASES = {'3dmaze': 'maze3d'}


def load_skin_description(assets_folder: str) -> Dict[str, str]:
    """
    从 assets_folder 加载 description.json 并返回 visual_description 字典。

    Args:
        assets_folder: 皮肤资源文件夹路径

    Returns:
        visual_description 字典

    Raises:
        FileNotFoundError: description.json 不存在
        ValueError: JSON 解析失败或缺少 visual_description
    """
    description_path = Path(assets_folder) / "description.json"

    if not description_path.exists():
        raise FileNotFoundError(f"description.json not found in {assets_folder}")

    try:
        with open(description_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse description.json in {assets_folder}: {e}")

    visual_desc = data.get("visual_description")
    if not visual_desc:
        raise ValueError(f"visual_description not found in {description_path}")

    return visual_desc


def get_dynamic_prompt(game_name: str, prompt_type: str, assets_folder: str) -> str:
    """
    获取动态替换后的 prompt。

    Args:
        game_name: 游戏类型 (maze, sokoban, trapfield, pathfinder, maze3d)
        prompt_type: prompt 类型 ('system' 或 'user')
        assets_folder: 皮肤资源文件夹路径

    Returns:
        格式化后的 prompt 字符串

    Raises:
        ValueError: 游戏类型/prompt类型未知，或皮肤描述缺少必需键
        FileNotFoundError: description.json 不存在
    """
    # 解析别名
    game_name = GAME_ALIASES.get(game_name, game_name)

    if game_name not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown game: {game_name}")
    if prompt_type not in PROMPT_TEMPLATES[game_name]:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

    template = PROMPT_TEMPLATES[game_name][prompt_type]
    visual_desc = load_skin_description(assets_folder)

    try:
        return template.format(**visual_desc)
    except KeyError as e:
        raise ValueError(f"Missing key in visual_description: {e}")

