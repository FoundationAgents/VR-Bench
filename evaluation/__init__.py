"""评估系统 - 包含视频模型评估和VLM评估"""
import importlib
from typing import TYPE_CHECKING

from . import videomodel_eval

if TYPE_CHECKING:  # 仅供类型检查，运行时不导入
    from . import vlm_eval

# 从videomodel_eval导出
from evaluation.videomodel_eval.extractor import CSRTTracker
from evaluation.videomodel_eval.evaluator import TrajectoryEvaluator
from evaluation.videomodel_eval.metrics import (
    PrecisionRateMetric,
    StepMetric,
    ExactMatchMetric,
    normalize_trajectory,
    resample_by_length,
    compute_path_length
)

__all__ = [
    'vlm_eval',
    'videomodel_eval',
    'CSRTTracker',
    'TrajectoryEvaluator',
    'PrecisionRateMetric',
    'StepMetric',
    'ExactMatchMetric',
    'normalize_trajectory',
    'resample_by_length',
    'compute_path_length'
]


def __getattr__(name):
    """vlm_eval 需要 openai SDK，延迟导入。
    否则只做视频模型评估的用户没装 openai 就 import 不了本包。"""
    if name == 'vlm_eval':
        return importlib.import_module('.vlm_eval', __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")