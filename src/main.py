# reconciliation_service.py

import base64
import random
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from enum import Enum

# --- Вспомогательные классы и перечисления (Data Models and Enums) ---

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

# --- Основной класс сервиса (Service Class) ---

class ReconciliationAPIService:
    """
    Сервис для обработки актов сверки согласно описанному API.
    Предоставляет методы, соответствующие эндпоинтам API.
    """
    def __init__(self):
        # Хранилище для текущих процессов обработки.
        # Ключ - process_id (строка), значение - InternalProcessData.
        self._active_processes: Dict[str, InternalProcessData] = {}

    def _generate_process_id(self) -> str:
        """Генерирует уникальный идентификатор процесса."""
        return str(uuid.uuid4())

    def _simulate_document_initial_processing(self, process_data: InternalProcessData):
        """
        Имитирует первоначальную обработку (парсинг) PDF документа.
        В реальном приложении здесь будет сложная логика работы с PDF и извлечения данных.
        Эта функция вызывается асинхронно или в фоновом режиме после send_reconciliation_act.
        """
        # Имитация задержки на обработку
        time.sleep(random.uniform(1.0, 3.0)) # Варьируем время обработки

        # Случайным образом определяем исход обработки (успех/ошибка)
        # Вероятности: 80% успех, 20% ошибка
        if random.random() < 0.80:
            process_data.status_enum = ProcessStatusEnum.DONE
            process_data.seller = "ОАО \"Рога и Копыта\" (Имитация)"
            process_data.buyer = "ИП \"Лучший Покупатель\" (Имитация)"
            process_data.period = Period(from_date="01.05.2024", to_date="31.05.2024")
            process_data.debit_seller = [
                ActEntry(row_id=1, record="Продажа товара А", value=1500.75, date="05.05.2024"),
                ActEntry(row_id=3, record="Услуга Б", value=750.00, date="15.05.2024")
            ]
            process_data.credit_seller = [
                ActEntry(row_id=5, record="Возврат средств", value=100.00, date="20.05.2024")
            ]
            process_data.error_message_detail = None
        else:
            process_data.status_enum = ProcessStatusEnum.ERROR
            process_data.error_message_detail = "Имитация: Ошибка при автоматическом разборе документа PDF."
            # Остальные поля (seller, buyer и т.д.) остаются None или пустыми

    # --- Методы API ---

    def send_reconciliation_act(self, document_base64: str) -> Dict[str, Any]:
        """
        API: send_reconciliation_act
        Принимает документ в base64, инициирует его обработку и возвращает process_id.

        Args:
            document_base64 (str): Документ акта сверки в формате base64.

        Returns:
            Dict[str, Any]: Словарь с 'process_id' или сообщение об ошибке.
                            Ожидаемый HTTP статус при успехе: 200 OK (согласно API).
        """
        if not isinstance(document_base64, str) or not document_base64:
            # Это должно обрабатываться на уровне веб-фреймворка как HTTP 400 Bad Request
            # Здесь мы просто возвращаем ошибку для демонстрации
            return {"error": "Некорректные входные данные: 'document' должен быть непустой строкой base64."}

        process_id = self._generate_process_id()
        process_data = InternalProcessData(
            process_id=process_id,
            original_document_b64=document_base64
            # Статус по умолчанию PROCESSING
        )
        self._active_processes[process_id] = process_data

        # В реальном приложении здесь запускается фоновая задача для _simulate_document_initial_processing
        # Например, через Celery, RQ, или asyncio.create_task
        # print(f"DEBUG: Запущена имитация обработки для process_id: {process_id}")
        # Для простоты симуляции, можно вызвать ее здесь, но это сделает запрос синхронным.
        # Оставим статус PROCESSING, чтобы get_process_status мог показать "wait".
        # _simulate_document_initial_processing(process_data) # Если бы это было синхронно

        return {"process_id": process_id}

    def get_process_status(self, process_id: str) -> Dict[str, Any]:
        """
        API: process_status
        Возвращает статус обработки документа по его process_id.

        Args:
            process_id (str): Идентификатор процесса обработки.

        Returns:
            Dict[str, Any]: JSON-ответ в соответствии с API.
                            HTTP статус зависит от результата (200, 201, 404, 500).
        """
        process_data = self._active_processes.get(process_id)

        if not process_data:
            # HTTP статус: 404 Not Found
            return {"status": ProcessStatusEnum.NOT_FOUND.value, "message": "not found"}

        # Имитация того, что фоновая обработка могла завершиться к этому моменту
        if process_data.status_enum == ProcessStatusEnum.PROCESSING:
            # В реальной системе статус обновляется фоновым процессом.
            # Здесь мы можем симулировать это, если прошло достаточно времени или по случайности.
            # Для демонстрации, если процесс "висит" в обработке, есть шанс, что он завершится.
            if random.random() < 0.6: # 60% шанс, что обработка завершилась (успешно или с ошибкой)
                # print(f"DEBUG: Имитация завершения обработки для process_id: {process_id} при запросе статуса.")
                self._simulate_document_initial_processing(process_data)


        # Формируем ответ в зависимости от текущего статуса
        current_status_enum = process_data.status_enum

        if current_status_enum == ProcessStatusEnum.PROCESSING:
            # HTTP статус: 201 Created (согласно API)
            return {"status": current_status_enum.value, "message": "wait"}

        elif current_status_enum == ProcessStatusEnum.DONE:
            # HTTP статус: 200 OK
            response_data = {
                "status": current_status_enum.value,
                "message": "done",
                "seller": process_data.seller,
                "buyer": process_data.buyer,
                "period": process_data.period.to_dict() if process_data.period else None,
                "debit": [entry.to_dict() for entry in process_data.debit_seller],
                "credit": [entry.to_dict() for entry in process_data.credit_seller]
            }
            return response_data

        elif current_status_enum == ProcessStatusEnum.ERROR:
            # HTTP статус: 500 Internal Server Error
            return {
                "status": current_status_enum.value,
                "message": process_data.error_message_detail or "Произошла неизвестная ошибка при обработке."
            }
        
        # Этот случай не должен достигаться при правильной логике
        return {"status": -99, "message": "Неизвестное состояние процесса."}


    def fill_reconciliation_act(
        self,
        process_id: str,
        debit_data: List[Dict[str, Any]], # Данные по дебету от покупателя
        credit_data: List[Dict[str, Any]] # Данные по кредиту от покупателя
    ) -> Dict[str, Any]:
        """
        API: fill_reconciliation_act
        Принимает данные от покупателя для заполнения акта сверки и возвращает заполненный документ.

        Args:
            process_id (str): Идентификатор процесса.
            debit_data (List[Dict[str, Any]]): Список записей по дебету от покупателя.
            credit_data (List[Dict[str, Any]]): Список записей по кредиту от покупателя.

        Returns:
            Dict[str, Any]: JSON-ответ с заполненным документом в base64 или сообщение об ошибке.
                            HTTP статус зависит от результата (200, 201, 404, 500).
        """
        process_data = self._active_processes.get(process_id)

        if not process_data:
            # HTTP статус: 404 Not Found
            return {"message": "not found", "status_code_hint": 404} # Добавим подсказку для HTTP

        current_status_enum = process_data.status_enum

        if current_status_enum == ProcessStatusEnum.PROCESSING:
            # HTTP статус: 201 (согласно API)
            return {"message": "wait", "status_code_hint": 201}

        if current_status_enum == ProcessStatusEnum.ERROR:
            # HTTP статус: 500
            return {"message": process_data.error_message_detail or "Невозможно заполнить документ из-за предыдущей ошибки.",
                    "status_code_hint": 500}

        if current_status_enum != ProcessStatusEnum.DONE:
            # Общий случай, если документ не готов к заполнению (например, если бы были другие статусы)
            # HTTP статус: 400 Bad Request или 409 Conflict
            return {"message": "Документ не готов к заполнению (не в статусе 'done').",
                    "status_code_hint": 400}

        # Валидация и преобразование входных данных debit_data и credit_data
        try:
            process_data.debit_buyer = [ActEntry.from_dict(item) for item in debit_data]
            process_data.credit_buyer = [ActEntry.from_dict(item) for item in credit_data]
        except (TypeError, KeyError) as e:
            # HTTP статус: 400 Bad Request (неверный формат данных)
            return {"message": f"Неверный формат данных для записей дебета/кредита: {e}",
                    "status_code_hint": 400}

        # Имитация процесса заполнения документа данными покупателя
        # В реальном приложении здесь будет логика модификации PDF (например, используя `original_document_b64`
        # и данные `debit_buyer`, `credit_buyer`) и генерации нового base64.
        time.sleep(random.uniform(0.5, 1.5)) # Имитация работы

        # Создаем "заполненный" документ (простая имитация)
        # Объединяем исходный документ с данными покупателя в текстовом виде и кодируем в base64
        filled_content_str = (
            f"ИСХОДНЫЙ ДОКУМЕНТ:\n{base64.b64decode(process_data.original_document_b64).decode('utf-8', errors='ignore')}\n\n"
            f"ДАННЫЕ ПОКУПАТЕЛЯ (ДЕБЕТ):\n{process_data.debit_buyer}\n\n"
            f"ДАННЫЕ ПОКУПАТЕЛЯ (КРЕДИТ):\n{process_data.credit_buyer}\n\n"
            f"--- КОНЕЦ ЗАПОЛНЕННОГО ДОКУМЕНТА ---"
        )
        process_data.filled_document_b64 = base64.b64encode(filled_content_str.encode('utf-8')).decode('utf-8')

        # HTTP статус: 200 OK
        return {"document": process_data.filled_document_b64}

# --- Пример использования сервиса (для демонстрации и тестирования) ---
if __name__ == "__main__":
    service = ReconciliationAPIService()

    print("--- Демонстрационный сценарий работы сервиса ---")

    # 1. Отправка акта на обработку (send_reconciliation_act)
    print("\nШаг 1: Отправка акта на обработку...")
    # Генерируем простой base64 для примера
    sample_pdf_content = "Это содержимое исходного PDF документа для акта сверки."
    sample_pdf_base64 = base64.b64encode(sample_pdf_content.encode('utf-8')).decode('utf-8')

    send_response = service.send_reconciliation_act(document_base64=sample_pdf_base64)
    print(f"Ответ от send_reconciliation_act: {send_response}")
    
    process_id = send_response.get("process_id")
    if not process_id:
        print("Не удалось получить process_id, демонстрация прервана.")
        exit()

    # 2. Проверка статуса обработки (process_status) - несколько попыток
    print(f"\nШаг 2: Проверка статуса для process_id: {process_id} (до 10 попыток)")
    status_response = None
    for attempt in range(10):
        print(f"  Попытка {attempt + 1}...")
        status_response = service.get_process_status(process_id=process_id)
        print(f"  Ответ от get_process_status: {status_response}")
        
        current_api_status = status_response.get("status")
        if current_api_status == ProcessStatusEnum.DONE.value:
            print("  Статус: Документ успешно обработан.")
            break
        elif current_api_status == ProcessStatusEnum.ERROR.value:
            print("  Статус: Ошибка при обработке документа.")
            break
        elif current_api_status == ProcessStatusEnum.PROCESSING.value:
            print("  Статус: Документ еще в обработке (wait). Ожидание...")
            time.sleep(1.2) # Пауза перед следующей попыткой
        elif current_api_status == ProcessStatusEnum.NOT_FOUND.value:
            print("  Статус: Процесс не найден (ошибка).")
            break
        else:
            print(f"  Неизвестный статус: {current_api_status}")
            break
    else: # Если цикл завершился без break (все попытки исчерпаны)
        print("  Документ не перешел в состояние DONE или ERROR за 10 попыток.")


    # 3. Заполнение акта сверки (fill_reconciliation_act) - если предыдущий шаг успешен
    if status_response and status_response.get("status") == ProcessStatusEnum.DONE.value:
        print(f"\nШаг 3: Заполнение акта сверки для process_id: {process_id}")
        
        # Пример данных от покупателя
        buyer_debit_entries = [
            {"row_id": 101, "record": "Оплата по счету #123 (покупатель)", "value": 1500.75, "date": "10.05.2024"},
        ]
        buyer_credit_entries = [
            {"row_id": 202, "record": "Аванс за услуги (покупатель)", "value": 500.00, "date": "12.05.2024"}
        ]

        fill_response = service.fill_reconciliation_act(
            process_id=process_id,
            debit_data=buyer_debit_entries,
            credit_data=buyer_credit_entries
        )
        
        if "document" in fill_response:
            print("  Акт успешно заполнен!")
            # print(f"  Ответ от fill_reconciliation_act (сокращенный base64): {{'document': '{fill_response['document'][:60]}...'}}")
            try:
                decoded_filled_doc = base64.b64decode(fill_response["document"]).decode('utf-8')
                print(f"  Расшифрованное (имитированное) содержимое заполненного документа:\n{'-'*20}\n{decoded_filled_doc}\n{'-'*20}")
            except Exception as e:
                print(f"  Не удалось расшифровать base64 заполненного документа: {e}")
        else:
            print(f"  Ошибка при заполнении акта: {fill_response}")

    elif status_response and status_response.get("status") == ProcessStatusEnum.ERROR.value:
        print(f"\nШаг 3 (пропущен): Заполнение акта невозможно, так как первоначальная обработка завершилась с ошибкой.")

    print("\n--- Демонстрация завершена ---")

    # Дополнительные тесты для других сценариев
    print("\n--- Дополнительный тест: Запрос статуса несуществующего процесса ---")
    non_existent_id = "id-которого-нет"
    status_non_existent = service.get_process_status(non_existent_id)
    print(f"Статус для ID '{non_existent_id}': {status_non_existent}")

    print("\n--- Дополнительный тест: Попытка заполнить несуществующий процесс ---")
    fill_non_existent = service.fill_reconciliation_act(non_existent_id, [], [])
    print(f"Заполнение для ID '{non_existent_id}': {fill_non_existent}")

