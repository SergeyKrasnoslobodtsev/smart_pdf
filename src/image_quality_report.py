from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class IssueLevel(Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    NOT_USABLE = 4

class IssueType(Enum):
    BLUR = ("blur", "Размытое изображение")
    OVEREXPOSURE = ("overexposure", "Передержка")
    UNDEREXPOSURE = ("underexposure", "Недодержка")
    NOISE_SALT_PEPPER = ("noise_sp", "Шум 'соль и перец'")
    NOISE_GAUSSIAN = ("noise_gauss", "Гауссов шум")
    LOW_CONTRAST = ("low_contrast", "Низкий контраст")

    def __init__(self, code, description):
        self._value_ = code
        self.description = description

@dataclass
class ImageIssue:
    type: IssueType
    level: IssueLevel
    recommendation: str
    details: Optional[Dict[str, Any]] = None

    @property
    def is_critical(self) -> bool:
        return self.level in {IssueLevel.HIGH, IssueLevel.NOT_USABLE}
    
    @classmethod
    def create(
        cls,
        type: IssueType,
        level: IssueLevel,
        recommendation: str,
        details: Optional[Dict[str, Any]] = None
    ) -> "ImageIssue":
        return cls(type=type, level=level, recommendation=recommendation, details=details)



@dataclass
class ImageQualityReport:
    metrics: Dict[str, float]
    image_size: Tuple[int, int]
    issues: List["ImageIssue"] = field(default_factory=list)
    has_critical_issues: bool = False

    def add_issue(
        self,
        type: IssueType,
        level: IssueLevel,
        recommendation: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        issue = ImageIssue.create(type, level, recommendation, details)
        self.issues.append(issue)
        if issue.is_critical:
            self.has_critical_issues = True

    def get_primary_issue(self) -> Optional[ImageIssue]:
        if not self.issues:
            return None
        return max(self.issues, key=lambda x: x.level.value)

    def get_issues_by_level(self, level: IssueLevel) -> List[ImageIssue]:
        return [i for i in self.issues if i.level == level]

    def get_recommendations_summary(self) -> str:
        if not self.issues:
            return "✅ Изображение хорошего качества. Дополнительная обработка не требуется."
        result = []
        prefix_map = {
            IssueLevel.LOW: "🟢",
            IssueLevel.MEDIUM: "🟡",
            IssueLevel.HIGH: "🔴",
            IssueLevel.NOT_USABLE: "❌",
        }
        for issue in sorted(self.issues, key=lambda x: x.level.value, reverse=True):
            if issue.level == IssueLevel.NONE:
                continue
            prefix = prefix_map.get(issue.level, "")
            result.append(f"{prefix} {issue.type.description}: {issue.recommendation}")
        if any(i.is_critical for i in self.issues):
            result.append("⚠️  Имеются критические проблемы. Повторное сканирование обязательно!")
        return "\n".join(result)
    