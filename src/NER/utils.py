import logging
import re
import typing
import datetime

from pullenti.ner.ProcessorService import ProcessorService
from pullenti.ner.Referent import Referent
from pullenti.ner.SourceOfAnalysis import SourceOfAnalysis
from pullenti.ner.date.DateAnalyzer import DateAnalyzer
from pullenti.ner.date.DateRangeReferent import DateRangeReferent
from pullenti.ner.date.DateReferent import DateReferent


def get_quarter_end_date(year: int, quarter: int) -> typing.Optional[datetime.date]:
    """Возвращает последний день указанного квартала."""
    if not (1 <= quarter <= 4) or year <= 0:
        return None
    
    if quarter == 1:
        return datetime.date(year, 3, 31)
    elif quarter == 2:
        return datetime.date(year, 6, 30)
    elif quarter == 3:
        return datetime.date(year, 9, 30)
    elif quarter == 4:
        return datetime.date(year, 12, 31)
    return None

def extract_date_from_text(
    txt: str, 
    logger: logging.Logger, 
    context_year: typing.Optional[int] = None
) -> typing.Optional[dict]:
    """
    Извлекает дату из текста.
    Сначала пытается найти кварталы регулярным выражением.
    Затем использует Pullenti.
    Приоритеты:
    1. Конец квартала (из regex или Pullenti).
    2. Полная дата (дд.мм.гггг) из Pullenti.
    3. Месяц и год (мм.гггг) из Pullenti (день по умолчанию 1).
    4. День и месяц (дд.мм) из Pullenti + context_year.
    5. Только месяц (мм) из Pullenti (день по умолчанию 1) + context_year.
    """
    potential_dates_info = []
    if not txt:
        return None

    # --- Шаг 1: Поиск кварталов регулярным выражением ---
    # Паттерны для кварталов: "1 квартал", "1-й квартал", "I квартал", "1 кв." и т.д.
    # Учитываем возможные вариации написания.
    quarter_patterns = [
        r"(?P<quarter_num>[1-4РIIVX]{1,3})\s*[-й]*(?:й|ого|му|го|м)?\s*(?:кв\.|квартал[а-яё]*)\b",
        r"\b(?P<quarter_word>перв(?:ый|ого|ому)|втор(?:ой|ого|ому)|трет(?:ий|ьего|ьему)|четверт(?:ый|ого|ому))\s*[-й]*(?:й|ого|му|го|м)?\s*(?:кв\.|квартал[а-яё]*)\b"
    ]
    
    quarter_map_roman = {'i': 1, 'ii': 2, 'iii': 3, 'iv': 4}
    quarter_map_digit_suffix = {'1':1, '2':2, '3':3, '4':4, 
                                '1-й':1, '2-й':2, '3-й':3, '4-й':4,
                                '1й':1, '2й':2, '3й':3, '4й':4}
    quarter_map_words = {
        'перв': 1, 'втор': 2, 'трет': 3, 'четверт': 4
    }

    found_quarter_by_regex = False
    if context_year and context_year > 0: # Квартал без года не имеет смысла
        for pattern in quarter_patterns:
            match = re.search(pattern, txt, re.IGNORECASE)
            if match:
                q_num = 0
                if match.groupdict().get('quarter_num'):
                    q_str = match.group('quarter_num').lower()
                    q_num = quarter_map_roman.get(q_str) or quarter_map_digit_suffix.get(q_str) or (int(q_str) if q_str.isdigit() and 1 <= int(q_str) <= 4 else 0)
                elif match.groupdict().get('quarter_word'):
                    q_word_part = match.group('quarter_word').lower()[:4] # Берем первые 4 буквы для сопоставления
                    q_num = quarter_map_words.get(q_word_part, 0)

                if q_num > 0:
                    quarter_end_dt = get_quarter_end_date(context_year, q_num)
                    if quarter_end_dt:
                        potential_dates_info.append({
                            'day': quarter_end_dt.day, 
                            'month': quarter_end_dt.month, 
                            'year': quarter_end_dt.year, 
                            'type': 'quarter_end_regex'
                        })
                        logger.debug(f"Regex нашел квартал Q{q_num} для года {context_year} в '{txt}', дата: {quarter_end_dt.strftime('%d.%m.%Y')}")
                        found_quarter_by_regex = True # Если нашли квартал regex-ом, можем его приоритизировать
                        break # Достаточно одного совпадения квартала по regex
            if found_quarter_by_regex:
                break
    
    # --- Шаг 2: Анализ с помощью Pullenti (если regex не нашел квартал или для других типов дат) ---
    # Можно добавить условие not found_quarter_by_regex, если хотим, чтобы Pullenti не запускался,
    # если regex уже нашел квартал. Но Pullenti может найти более точную полную дату.
    # Поэтому оставим Pullenti работать всегда, но приоритет отдадим regex-кварталу, если он есть.

    try:
        with ProcessorService.create_specific_processor(DateAnalyzer.ANALYZER_NAME) as proc:
            analysis_result = proc.process(SourceOfAnalysis(txt))
            entities: typing.List[Referent] = analysis_result.entities
            
            logger.debug(f"Анализ текста '{txt}' дал {len(entities)} сущностей Pullenti.")
            for i, entity in enumerate(entities):
                # ... (остальная логика Pullenti из предыдущего ответа остается здесь) ...
                # Логика извлечения DateReferent и DateRangeReferent (включая кварталы от Pullenti)
                logger.debug(f"  Сущность {i}: {type(entity).__name__} - '{str(entity)}'")
                
                date_ref: typing.Optional[DateReferent] = None
                is_quarter_range_pullenti = False
                quarter_num_pullenti = 0
                quarter_year_candidate_pullenti = 0

                if isinstance(entity, DateRangeReferent):
                    logger.debug(f"    Pullenti DateRangeReferent: Q={entity.quarter}, From={entity.date_from}, To={entity.date_to}")
                    if entity.quarter > 0:
                        is_quarter_range_pullenti = True
                        quarter_num_pullenti = entity.quarter
                        if entity.date_to and entity.date_to.year > 0:
                            quarter_year_candidate_pullenti = entity.date_to.year
                        elif entity.date_from and entity.date_from.year > 0:
                            quarter_year_candidate_pullenti = entity.date_from.year
                        logger.debug(f"    Pullenti обнаружил квартал Q{quarter_num_pullenti}. Кандидат года: {quarter_year_candidate_pullenti}")
                    elif entity.date_from:
                        date_ref = entity.date_from
                elif isinstance(entity, DateReferent):
                    date_ref = entity
                
                if is_quarter_range_pullenti:
                    year_for_quarter_pullenti = quarter_year_candidate_pullenti if quarter_year_candidate_pullenti > 0 else context_year
                    if year_for_quarter_pullenti and year_for_quarter_pullenti > 0:
                        quarter_end_dt_pullenti = get_quarter_end_date(year_for_quarter_pullenti, quarter_num_pullenti)
                        if quarter_end_dt_pullenti:
                            potential_dates_info.append({
                                'day': quarter_end_dt_pullenti.day, 
                                'month': quarter_end_dt_pullenti.month, 
                                'year': quarter_end_dt_pullenti.year, 
                                'type': 'quarter_end_pullenti'
                            })
                            logger.debug(f"    Pullenti добавил дату конца квартала: {quarter_end_dt_pullenti} (год {year_for_quarter_pullenti})")
                elif date_ref:
                    p_day = date_ref.day if date_ref.day > 0 else 0
                    p_month = date_ref.month if date_ref.month > 0 else 0
                    p_year = date_ref.year if date_ref.year > 0 else 0

                    if p_month > 0:
                        if p_year > 0:
                            if p_day > 0:
                                potential_dates_info.append({'day': p_day, 'month': p_month, 'year': p_year, 'type': 'full_dmy_pullenti'})
                            else:
                                potential_dates_info.append({'day': 1, 'month': p_month, 'year': p_year, 'type': 'month_year_pullenti'})
                        else: 
                            if p_day > 0:
                                potential_dates_info.append({'day': p_day, 'month': p_month, 'year': None, 'type': 'day_month_only_pullenti'})
                            else:
                                potential_dates_info.append({'day': 1, 'month': p_month, 'year': None, 'type': 'month_only_pullenti'})

    except Exception as e_pullenti:
        logger.exception(f"Ошибка при обработке текста '{txt}' с Pullenti: {e_pullenti}")
        # Не прерываем выполнение, если Pullenti упал, regex мог уже что-то найти

    if not potential_dates_info:
        logger.debug(f"Для текста '{txt}' не найдено потенциальных дат (regex и Pullenti).")
        return None

    logger.debug(f"Все потенциальные даты для '{txt}': {potential_dates_info}")
    best_date_components = None

    # Приоритетный поиск
    # Отдаем приоритет кварталу, найденному regex, если он есть и год контекста был доступен.
    # Затем квартал от Pullenti, затем полные даты и т.д.
    order_of_preference = [
        'quarter_end_regex',      # Квартал, найденный регулярным выражением
        'quarter_end_pullenti',   # Квартал, найденный Pullenti
        'full_dmy_pullenti',      # Полная дата от Pullenti
        'month_year_pullenti'     # Месяц-год от Pullenti
    ]
    
    for date_type in order_of_preference:
        for pd_info in potential_dates_info:
            if pd_info['type'] == date_type:
                best_date_components = pd_info
                break
        if best_date_components:
            break
    
    if not best_date_components and context_year:
        context_order_preference = ['day_month_only_pullenti', 'month_only_pullenti']
        for date_type_ctx in context_order_preference:
            for pd_info in potential_dates_info:
                if pd_info['type'] == date_type_ctx and pd_info.get('year') is None: # Убедимся, что год еще не установлен
                    best_date_components = {
                        'day': pd_info['day'], 
                        'month': pd_info['month'], 
                        'year': context_year, 
                        # Обновляем тип для логгирования, чтобы показать, что использовался контекст
                        'type': pd_info['type'].replace('_pullenti', '_context').replace('_only', '_context') 
                    }
                    break
            if best_date_components:
                break
    
    if best_date_components and best_date_components.get('year'):
        final_day = best_date_components['day']
        final_month = best_date_components['month']
        final_year = best_date_components['year']
        
        formatted_str = f"{final_day:02d}.{final_month:02d}.{final_year:04d}"
        logger.debug(
            f"Извлечена дата: {formatted_str} из '{txt}' (тип: {best_date_components['type']})"
        )
        return {
            'day': final_day, 
            'month': final_month, 
            'year': final_year, 
            'formatted_str': formatted_str
        }
    else:
        logger.debug(f"Не удалось окончательно определить дату из '{txt}'. Лучший кандидат: {best_date_components}, Контекстный год: {context_year}")
        return None

def format_currency_value(value: str) -> str:
    text_to_process = value.strip()
    original_value_to_return = value

    if not any(char.isdigit() for char in text_to_process):
        return original_value_to_return

    numeric_part_str = text_to_process
    
    if any(char.isalpha() for char in numeric_part_str):
        return original_value_to_return

    s = numeric_part_str.replace(" ", "")
    
    if not s:
        return original_value_to_return

    s = s.replace(',', '.')

    if s.count('.') > 1:
        parts = s.split('.')
        integer_part = "".join(parts[:-1])
        decimal_part = parts[-1]
        s = f"{integer_part}.{decimal_part}"

    match = re.fullmatch(r"(-?\d+)(\.(\d{1,2}))?", s)

    if match:
        integer_part = match.group(1)
        decimal_digits_str = match.group(3)

        if decimal_digits_str:
            formatted_decimal = f"{decimal_digits_str}0" if len(decimal_digits_str) == 1 else decimal_digits_str
            return f"{integer_part},{formatted_decimal}"
        else:
            return f"{integer_part},00"
    else:
        return original_value_to_return