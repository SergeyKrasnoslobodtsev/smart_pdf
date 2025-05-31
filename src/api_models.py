from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.PDFExtractor.base_extractor import Document

class ProcessStatusEnum(Enum):
    """
    Перечисление для статусов обработки документа.
    Соответствует значениям 'status' в ответах API.
    """
    PROCESSING = 0  # В обработке (сообщение "wait")
    DONE = 1        # Успешно обработан (сообщение "done")
    NOT_FOUND = -1  # Не найден (сообщение "not found")
    ERROR = -2      # Ошибка обработки (сообщение содержит описание ошибки)

@dataclass
class ActEntry:
    """
    Представляет одну запись (строку) в акте сверки (дебет/кредит).
    Используется для данных в полях 'debit' и 'credit'.
    """
    row_id: int
    record: str
    value: float  # Сумма, указана как float в описании
    date: Optional[str] = None # Дата, может отсутствовать

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует объект в словарь для JSON-сериализации."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActEntry':
        """Создает объект из словаря."""
        return cls(**data)

@dataclass
class Period:
    """
    Представляет период дат (с ... по ...).
    Используется для поля 'period'.
    """
    from_date: str # В API это ключ "from"
    to_date: str   # В API это ключ "to"

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует объект в словарь, адаптируя ключи для API ('from', 'to')."""
        return {"from": self.from_date, "to": self.to_date}

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'Period':
        """Создает объект из словаря, ожидая ключи 'from' и 'to'."""
        return cls(from_date=data["from"], to_date=data["to"])

@dataclass
class InternalProcessData:
    """
    Внутреннее представление данных процесса обработки акта сверки.
    Этот класс не является частью API, а используется сервисом для хранения состояния.
    """
    process_id: str
    original_document_b64: str  # Исходный документ, полученный от пользователя
    status_enum: ProcessStatusEnum = ProcessStatusEnum.PROCESSING # Текущий статус обработки
    # Поля, заполняемые после успешного парсинга документа (этап process_status -> done)
    seller: Optional[str] = None
    buyer: Optional[str] = None
    period: Optional[Period] = None
    debit_seller: List[ActEntry] = field(default_factory=list)  # Данные продавца
    credit_seller: List[ActEntry] = field(default_factory=list) # Данные продавца
    # Поля, заполняемые на этапе fill_reconciliation_act (данные от покупателя)
    debit_buyer: List[ActEntry] = field(default_factory=list)
    credit_buyer: List[ActEntry] = field(default_factory=list)
    # Заполненный документ (результат fill_reconciliation_act)
    filled_document_b64: Optional[str] = None
    # Сообщение об ошибке, если status_enum == ERROR
    error_message_detail: Optional[str] = None