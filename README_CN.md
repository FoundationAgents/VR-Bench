<div align="center">

# VR-Bench: 视觉语言模型的视觉推理基准测试

</div>

<div align="center" style="font-size: 15pt">

<a href='https://arxiv.org/abs/TBD'><img src='https://img.shields.io/badge/Arxiv-TBD-purple'></a>
<a href='https://huggingface.co/papers/TBD'><img src='https://img.shields.io/badge/HF%20Paper-TBD-blue'></a>
<a href='https://github.com/SNHuan/VR-Bench'><img src='https://img.shields.io/badge/Project-Website-green'></a>
<a href='https://huggingface.co/datasets/amagipeng/VR-Bench'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow'></a>
<a href='https://huggingface.co/datasets/HY-Wan/VR-Bench/tree/main'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange'></a>

</div>

中文文档 | [English](README.md)

VR-Bench 是一个综合性的基准测试框架，用于评估视觉语言模型（VLMs）在空间推理和规划任务上的能力。通过多种益智游戏，提供统一的数据集生成、评估和分析框架�?

## 🧩 基准概览

VR-Bench 总览。（A）迷宫类型：VR-Bench 包含五种迷宫——规则迷宫、不规则迷宫、3D 迷宫、Trapfield 和 Sokoban，覆盖 2D 与 3D 场景以及多样的任务结构，提供丰富的空间推理场景。（B）视频推理范式：VR-Bench 采用链式帧推理范式，要求模型对视频逐帧进行推理，从而捕捉顺序视觉推理能力。（C）基准表现：我们在所有迷宫类型上，基于四个核心指标评估主流 VLM 和视频模型，揭示其空间推理能力的差异。（D）附加分析：VR-Bench 还支持对难度泛化、纹理泛化、迷宫类型泛化以及测试时扩展进行评估，以全面衡量模型的鲁棒性与泛化能力。

![video reason](./resource/video_reason.svg)

为评估 VTR 任务上的泛化能力并提升在多样迷宫场景中的鲁棒性，我们从两个关键维度进行变化：（1）**难度等级**：通过调整迷宫规模（例如从 5×5 扩展到 7×7）、修改迷宫分支数量以及增加障碍等方式，定义简单、中等和困难三个等级；（2）**迷宫纹理**：基于程序化方法和生成模型，改变迷宫中障碍、路径及其他组件的纹理，使策略暴露于更广泛的视觉分布，从而缓解对干净、合成环境的过拟合。

![variant](./resource/variant.svg)

## 🎮 支持的游�?

VR-Bench 包含五种不同的益智游戏，每种游戏测试视觉推理的不同方面：

- **Maze（迷宫）**: 在网格迷宫中从起点导航到终点
- **Sokoban（推箱子�?*: 将箱子推到目标位置，同时避开墙壁
- **3D Maze�?D迷宫�?*: 多层迷宫，通过梯子连接不同楼层
- **PathFinder（路径查找）**: 在带有标记路径点的不规则迷宫中寻找路�?
- **TrapField（陷阱场�?*: 在场地中导航，同时避开陷阱

## �?核心特�?

- **程序化生�?*: 自动生成多样化的关卡，支持可配置的难度等�?
- **纹理自定�?*: 通过纹理皮肤支持自定义视觉主�?
- **视频渲染**: 生成流畅的解决方案动画视频（24 FPS�?
- **VLM评估**: 内置框架支持测试各种VLM（GPT、Gemini、Qwen等）
- **全面指标**: 成功率（SR）、路径比率（PR）、移动比率（MR�?
- **并行处理**: 多线程生成和评估，提高效�?
- **去重机制**: 自动检测和移除重复关卡

## 📋 环境要求

- Python >= 3.10
- CUDA兼容的GPU（可选，用于本地VLM推理�?

## 🚀 快速开�?

### 1. 安装

```bash
# 克隆仓库
git clone https://github.com/SNHuan/VR-Bench.git
cd VR-Bench

# 安装依赖
pip install -r requirements.txt
```

### 2. 环境配置

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，配置：
# - VLM评估所需的API密钥
# - 数据集路�?
# - CUDA配置
```

### 3. 下载数据�?

```bash
# �?Hugging Face 下载预生成的数据�?
python dataset_init.py --output-dir ./dataset_VR
```

### 4. 生成自定义关�?

```bash
# 编辑 config/config.yaml 配置游戏类型和难�?
# 然后运行批量生成
python -m generation.batch_generate config/config.yaml
```

### 5. 评估VLM

```bash
# 启动本地VLM服务器（可选，用于本地模型�?
bash scripts/start_sglang_server.sh

# 运行评估
bash scripts/run_vlm_eval.sh
```

## 📁 项目结构

```
VR-Bench/
├── core/                   # 核心框架
�?  ├── schema/            # 统一状态表�?
�?  ├── renderer.py        # 基础渲染引擎
�?  ├── texture_handler.py # 纹理管理
�?  └── game_adapter.py    # 游戏适配器接�?
├── games/                 # 游戏实现
�?  ├── maze/             # 迷宫游戏
�?  ├── sokoban/          # 推箱子游�?
�?  ├── maze3d/           # 3D迷宫游戏
�?  ├── pathfinder/       # 路径查找游戏
�?  └── trapfield/        # 陷阱场游�?
├── generation/           # 数据集生�?
�?  ├── batch_generate.py # 批量生成工具
�?  └── generate_videos.py # 视频生成
├── evaluation/           # VLM评估
�?  └── vlm_eval/        # 评估框架
├── config/              # 配置文件
�?  ├── config.yaml      # 生成配置
�?  └── vlm/            # 评估配置
├── skins/              # 纹理资源
└── scripts/            # 实用脚本
```

## 🎯 使用示例

### 生成迷宫数据�?

```bash
# 编辑 config/config.yaml
game_type: "maze"
skins_root: "skins/maze"
difficulties:
  small:
    maze_size: 9
    count: 100

# 运行生成
python -m generation.batch_generate config/config.yaml
```

### 在推箱子游戏上评�?

```bash
# 编辑 config/vlm/sokoban_eval.yaml
# 配置模型和数据集路径

# 运行评估
python -m evaluation.vlm_eval.run_vlm_eval config/vlm/sokoban_eval.yaml
```

## 📊 评估指标

- **成功率（SR）**: 正确解决关卡的百分比
- **路径比率（PR）**: 从起点开始连续正确动作的比率
- **移动比率（MR）**: 解答是否与参考解完全一致的二值指标
- **步数统计**: 解答中动作的总步数

## 🔧 配置说明

### 生成配置 (`config/config.yaml`)

- `game_type`: 要生成的游戏类型（maze、sokoban、pathfinder、trapfield、maze3d）
- `skins_root`: 纹理资源路径
- `difficulties`: 难度等级及参数
- `generation.max_attempts`: 生成有效关卡的最大尝试次数
- `parallel.max_workers`: 并行工作进程数

### 评估配置 (`config/vlm/*.yaml`)

- `game`: 要评估的游戏类型
- `dataset`: 数据集路�?
- `models`: 要测试的VLM列表
- `workers`: 并行评估工作进程�?
- `max_levels`: 最大评估关卡数�?1表示全部�?

## 🎨 自定义纹�?

每个游戏都支持自定义纹理皮肤�?

1. �?`skins/<game_name>/` 下创建新文件�?
2. 添加所需的纹理图片（PNG/JPG格式�?
3. 在配置文件中指定皮肤路径

所需纹理文件因游戏而异，请参考现有皮肤文件夹�?

### 各游戏纹理要�?

- **Maze**: wall, floor, player, goal
- **Sokoban**: wall, floor, player, box, target
- **PathFinder**: 自定义背景和路径纹理
- **TrapField**: floor, trap, player, goal

## 🔬 扩展新游�?

VR-Bench 使用适配器模式，便于添加新游戏：

1. �?`games/` 下创建新游戏目录
2. 实现 `GameAdapter` 接口�?
   - `generate_level()`: 关卡生成逻辑
   - `save_level()`: 保存关卡数据和渲染输�?
   - `get_level_hash()`: 用于去重
   - `is_duplicate()`: 重复检�?
3. 实现游戏特定逻辑和渲�?
4. �?`evaluation/vlm_eval/executors/` 创建执行�?
5. �?`generation/batch_generate.py` 中注�?

详细说明请参考现有游戏实现�?

## 🐛 常见问题

### 问题排查

**问题**: VLM推理时CUDA内存不足
- **解决方案**: 减小批处理大小或使用多GPU张量并行

**问题**: 视频生成失败
- **解决方案**: 确保已安装ffmpeg：`pip install imageio-ffmpeg`

**问题**: API速率限制
- **解决方案**: 减少评估配置中的 `workers` 数量或添加延�?

**问题**: 生成重复关卡
- **解决方案**: 增加生成配置中的 `max_duplicate_retries`

**问题**: 纹理加载失败
- **解决方案**: 检查纹理文件格式（支持PNG/JPG）和路径配置

## 💡 最佳实�?

### 数据集生�?

1. **从小规模开�?*: 先生成少量关卡测试配�?
2. **验证可解�?*: 确保 `check_solvable: true`
3. **使用多皮�?*: 为同一游戏准备多个纹理皮肤增加多样�?
4. **合理设置难度**: 根据目标逐步增加难度参数

### VLM评估

1. **预热模型**: 首次运行前先测试API连接
2. **监控成本**: 使用本地模型或设�?`max_levels` 限制
3. **保存结果**: 评估结果自动保存�?`output_dir`
4. **批量测试**: 在配置文件中列出多个模型进行对比

## 📊 性能优化

- **并行生成**: 根据CPU核心数调�?`max_workers`
- **GPU利用**: 使用SGLang进行高效的本地VLM推理
- **缓存模型**: 设置 `HF_CACHE_DIR` 避免重复下载
- **视频压缩**: 调整FPS和分辨率平衡质量与文件大�?

## 📚 引用

如果您在研究中使用了 VR-Bench，请引用�?

```bibtex
@misc{vrbench2025,
  title={VR-Bench: Visual Reasoning Benchmark for Vision-Language Models},
  author={VR-Bench Team},
  year={2025},
  url={https://github.com/SNHuan/VR-Bench}
}
```

## 🤝 贡献

欢迎贡献！请随时提交 Pull Request。对于重大更改：

1. Fork 本仓�?
2. 创建特性分�?(`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开�?Pull Request

### 贡献指南

- 遵循现有代码风格
- 添加适当的注释和文档
- 确保所有测试通过
- 更新相关文档

## 🔗 相关资源

- [Hugging Face Dataset](https://huggingface.co/datasets/amagipeng/VR-Bench)

## 📝 许可�?

本项目采�?MIT 许可�?- 详见 LICENSE 文件�?

## 🙏 致谢

VR-Bench 基于多个开源项目和视觉推理、VLM评估领域的研究成果�?

## 📧 联系方式

如有问题和反馈，请在 GitHub 上提�?issue 或联系维护者�?

