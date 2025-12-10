#!/usr/bin/env python3
"""
使用 VLM 自动识别皮肤图片内容，生成 description.json 文件。

用法:
    python scripts/analyze_skins.py --skins-root skins --model gemini-2.5-flash-image
    python scripts/analyze_skins.py --skins-root skins --game maze --skin-id 1
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 支持的图片格式（与 core/constants.py 保持一致）
SUPPORTED_IMAGE_FORMATS = ('.png', '.jpg', '.jpeg')

# 每种游戏类型的皮肤元素（不包含 maze3d）
# 只列出元素名称，文件格式会自动检测
GAME_ELEMENTS = {
    'maze': ['player', 'target', 'wall', 'floor'],
    'sokoban': ['player', 'target', 'box', 'wall', 'floor'],
    'trapfield': ['player', 'goal', 'trap', 'floor'],
    'pathfinder': ['start', 'end', 'road']
}

# 元素名称到描述键的映射
ELEMENT_NAME_MAP = {
    'target': 'goal',  # maze 和 sokoban 中 target.png 对应 goal
}


def find_texture_file(skin_folder: Path, element_name: str) -> Optional[Path]:
    """查找纹理文件，支持多种格式"""
    for ext in SUPPORTED_IMAGE_FORMATS:
        candidate = skin_folder / f"{element_name}{ext}"
        if candidate.exists():
            return candidate
    return None

SYSTEM_PROMPT = """You are an expert at describing game sprite images.
Your task is to describe what this sprite image shows in a concise phrase.
Focus on:
1. The object/character type (e.g., ball, rabbit, treasure chest, stone wall, wooden crate)
2. The primary color (e.g., red, blue, golden, gray)
3. The shape if relevant (e.g., circle, square, star)

Return ONLY a short descriptive phrase, nothing else.
Examples: "red ball", "golden treasure chest", "gray stone bricks", "white rabbit", "green circle", "wooden floor tiles"
"""

USER_PROMPT = "Describe this game sprite image in a few words. What object is it and what color?"


def analyze_image(vlm_client, image_path: str) -> str:
    """使用 VLM 分析单张图片"""
    try:
        response = vlm_client.query(SYSTEM_PROMPT, USER_PROMPT, image_path)
        # 清理响应，去除引号和多余空白
        description = response.strip().strip('"\'').lower()
        return description
    except Exception as e:
        logging.error(f"Failed to analyze {image_path}: {e}")
        return "unknown"


def analyze_skin_folder(vlm_client, skin_folder: Path, game_type: str) -> Dict:
    """分析一个皮肤文件夹中的所有图片"""
    if game_type not in GAME_ELEMENTS:
        logging.warning(f"Unknown game type: {game_type}")
        return {}

    elements = GAME_ELEMENTS[game_type]
    visual_description = {}

    for element_name in elements:
        image_path = find_texture_file(skin_folder, element_name)
        if image_path:
            logging.info(f"  Analyzing {image_path.name}...")
            description = analyze_image(vlm_client, str(image_path))
            # 使用映射后的描述键（如 target -> goal）
            desc_key = ELEMENT_NAME_MAP.get(element_name, element_name)
            visual_description[desc_key] = description
            logging.info(f"    -> {desc_key}: {description}")
        else:
            logging.warning(f"  Image not found: {element_name}.* in {skin_folder}")

    return visual_description


def process_skin(vlm_client, skin_folder: Path, game_type: str, skin_id: str):
    """处理单个皮肤并生成 description.json"""
    logging.info(f"Processing {game_type}/skin_{skin_id}...")
    
    visual_description = analyze_skin_folder(vlm_client, skin_folder, game_type)
    
    if not visual_description:
        logging.warning(f"No elements found for {skin_folder}")
        return
    
    description_data = {
        "game_type": game_type,
        "skin_id": skin_id,
        "visual_description": visual_description
    }
    
    output_path = skin_folder / "description.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(description_data, f, indent=2, ensure_ascii=False)
    
    logging.info(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze skin images using VLM')
    default_skins_root = PROJECT_ROOT / 'skins'
    parser.add_argument('--skins-root', type=str, default=str(default_skins_root),
                        help=f'Root directory of skins (default: {default_skins_root})')
    parser.add_argument('--game', type=str, default="sokoban",
                        help='Only process specific game type (maze/sokoban/trapfield/pathfinder)')
    parser.add_argument('--skin-id', type=str, default=None,
                        help='Only process specific skin ID (e.g., 1, 2, 3)')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='VLM model to use (default: gpt-4o)')
    parser.add_argument('--base-url', type=str, default=None,
                        help='API base URL (default: from env or OpenAI)')
    parser.add_argument('--api-key', type=str, default=None,
                        help='OpenAI API key (default: from env OPENAI_API_KEY)')
    parser.add_argument('--dry-run', action='store_true',
                        help='List files without actually calling VLM')
    args = parser.parse_args()
    
    skins_root = Path(args.skins_root)
    if not skins_root.exists():
        logging.error(f"Skins root not found: {skins_root}")
        return
    
    # 初始化 VLM 客户端
    if not args.dry_run:
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)
        except ImportError:
            pass

        from evaluation.vlm_eval.vlm_client import VLMClient

        api_key = args.api_key or os.getenv('OPENAI_API_KEY')
        base_url = args.base_url or os.getenv('OPENAI_BASE_URL')

        if not api_key:
            logging.error("API key not provided. Use --api-key or set OPENAI_API_KEY in .env")
            logging.info("You can copy .env.example to .env and fill in your API key")
            return

        vlm_client = VLMClient(
            model=args.model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=100,
            temperature=0.0
        )
    else:
        vlm_client = None
    
    # 确定要处理的游戏类型
    game_types = [args.game] if args.game else list(GAME_ELEMENTS.keys())
    
    for game_type in game_types:
        game_folder = skins_root / game_type
        if not game_folder.exists():
            logging.warning(f"Game folder not found: {game_folder}")
            continue
        
        # 获取所有皮肤文件夹
        skin_folders = sorted([d for d in game_folder.iterdir() if d.is_dir()])
        
        for skin_folder in skin_folders:
            skin_id = skin_folder.name
            
            # 如果指定了 skin_id，只处理该皮肤
            if args.skin_id and skin_id != args.skin_id:
                continue
            
            if args.dry_run:
                logging.info(f"[DRY RUN] Would process: {game_type}/{skin_id}")
                for element_name in GAME_ELEMENTS.get(game_type, []):
                    img_path = find_texture_file(skin_folder, element_name)
                    if img_path:
                        logging.info(f"  {element_name}: {img_path.name}")
                    else:
                        logging.info(f"  {element_name}: MISSING")
            else:
                process_skin(vlm_client, skin_folder, game_type, skin_id)


if __name__ == '__main__':
    main()

