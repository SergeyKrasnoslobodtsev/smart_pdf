from abc import ABC
from dataclasses import dataclass
import os
import yaml
from typing import Dict, Type, TypeVar, Any

@dataclass
class BlurValues:
    score_blurredness: float
@dataclass
class OverExposureValues:
    score_contrast: float
    score_luminosity: float 
@dataclass
class UnderExposureValues:
    score_contrast: float
    score_luminosity: float 

@dataclass
class NoiseValues:
    sharpness: float
    ssim: float = 0.0
    entropy: float = 0.0
    dark_ratio: float = 0.0
    contrast: float = 0.0

@dataclass
class ContrastValues:
    contrast_ratio: float
    dark_ratio: float
    light_ratio: float

@dataclass
class LevelConfigBase(ABC):
    message: str

@dataclass
class BlurLevelConfig(LevelConfigBase):
    values: BlurValues

@dataclass
class NoiseLevelConfig(LevelConfigBase):
    values: NoiseValues

@dataclass
class OverExposureLevelConfig(LevelConfigBase):
    values: OverExposureValues

@dataclass
class UnderExposureLevelConfig(LevelConfigBase):
    values: UnderExposureValues

@dataclass
class ContrastLevelConfig(LevelConfigBase):
    values: ContrastValues

@dataclass
class BaseConfig(ABC):
    high: LevelConfigBase
    medium: LevelConfigBase
    low: LevelConfigBase
    not_usable: LevelConfigBase

@dataclass
class BlurConfig(BaseConfig):
    high: BlurLevelConfig
    medium: BlurLevelConfig
    low: BlurLevelConfig
    not_usable: BlurLevelConfig

@dataclass
class NoiseTypeConfig(BaseConfig):
    high: NoiseLevelConfig
    medium: NoiseLevelConfig
    low: NoiseLevelConfig
    not_usable: NoiseLevelConfig

@dataclass
class NoiseConfig:
    salt_pepper: NoiseTypeConfig
    gaussian: NoiseTypeConfig

@dataclass
class OverExposureConfig(BaseConfig):
    high: OverExposureLevelConfig
    medium: OverExposureLevelConfig
    low: OverExposureLevelConfig
    not_usable: OverExposureLevelConfig

@dataclass
class UnderExposureConfig(BaseConfig):
    high: UnderExposureLevelConfig
    medium: UnderExposureLevelConfig
    low: UnderExposureLevelConfig
    not_usable: UnderExposureLevelConfig

@dataclass
class ContrastConfig(BaseConfig):
    high: ContrastLevelConfig
    medium: ContrastLevelConfig
    low: ContrastLevelConfig
    not_usable: ContrastLevelConfig

@dataclass
class ImageParamsConfig:
    dark_threshold: float
    light_threshold: float

@dataclass
class ImageQualityConfig:
    image_params: ImageParamsConfig
    blur: BlurConfig
    noise: NoiseConfig
    overexposure: OverExposureConfig
    underexposure: UnderExposureConfig
    contrast: ContrastConfig

T = TypeVar("T")

def _from_dict(cls: Type[T], d: Any) -> T:
    """
    Рекурсивно преобразует словарь в dataclass (поддерживает вложенность).
    """
    if not hasattr(cls, "__dataclass_fields__"):
        return d 
    kwargs = {}
    for field_name, field_info in cls.__dataclass_fields__.items(): 
        value = d.get(field_name)
        field_type = field_info.type
        if value is None:
            kwargs[field_name] = None
        elif hasattr(field_type, "__dataclass_fields__"):
            kwargs[field_name] = _from_dict(field_type, value)
        elif (getattr(field_type, "__origin__", None) is dict or
              getattr(field_type, "__origin__", None) is Dict):
            kwargs[field_name] = dict(value)
        else:
            kwargs[field_name] = value
    return cls(**kwargs)

def load_config(path: str, config_type: Type[T]) -> T:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Конфигурационный файл не найден: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _from_dict(config_type, raw)


if __name__ == "__main__":
    config = load_config("./config/quality_thresholds.yaml", ImageQualityConfig)
    print(config.blur.high.values.score_blurredness)
    print(config.noise.gaussian.high.values.entropy)
    print(config.overexposure.high.values.dark_ratio)
    print(config.underexposure.high.message)