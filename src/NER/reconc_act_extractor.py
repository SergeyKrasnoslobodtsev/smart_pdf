
import logging
import re
import typing
from venv import logger

from .utils import extract_date_from_text
from .utils import format_currency_value

from src.PDFExtractor.base_extractor import Cell, Document

class ReconciliationActExtractor:
    """
    Извлекает данные из таблиц актов сверки.
    """
    def __init__(self, doc: Document, logger: logging.Logger):
        self.doc = doc
        self.logger = logger

    def _find_debit_credit_columns_under_header(
            self, table_cells: list[Cell], 
            parent_row: int, parent_col: int, parent_colspan: int,
            debit_kw: str = "дебет", credit_kw: str = "кредит"
    ) -> tuple[int, int]:
        debit_idx, credit_idx = -1, -1
        sub_header_row = parent_row + 1
        for cell in table_cells:
            logger.debug(f'text cell: {cell.text} - {cell.row}:{cell.col}')
            if cell.row == sub_header_row and parent_col <= cell.col < (parent_col + parent_colspan):
                text_low = cell.text.lower().strip() if cell.text else ""
                if debit_kw in text_low and debit_idx == -1: 
                    debit_idx = cell.col
                if credit_kw in text_low and credit_idx == -1 and cell.col != debit_idx: 
                    credit_idx = cell.col
        return debit_idx, credit_idx

    def extract_for_seller(self, seller_info: dict) -> list[dict]:
        transactions_data = []
        seller_names = set()
        
        if sr := seller_info.get('str_repr'):
            sr_low = sr.lower()
            seller_names.add(sr_low)
            seller_names.add(sr_low.split(',')[0].strip())
            seller_names.update(m.strip() for m in re.findall(r'\(([^,)]+)', sr_low) if m.strip() and len(m.strip()) > 1)
        seller_names.update(cn.lower() for cn in seller_info.get('canonical_names', []))
        
        if raw_txt := seller_info.get('text'):
            raw_low = raw_txt.lower()
            core_raw = raw_low.split(',')[0].strip()
            seller_names.add(core_raw)
            if qm := re.search(r'["«“]([^"»”]+)["»”]', core_raw): 
                seller_names.add(qm.group(1).strip())
            seller_names.update(m.strip() for m in re.findall(r'\(([^,)]+)', raw_low) if m.strip() and len(m.strip()) > 1)
        
        sorted_seller_names = sorted([name for name in seller_names if name], key=len, reverse=True)
        self.logger.debug(f"Варианты имени продавца для поиска: {sorted_seller_names}")

        for tbl_idx, tbl in enumerate(self.doc.get_tables()):
            self.logger.info(f"Анализ таблицы {tbl_idx + 1} для акта сверки продавца.")
            main_hdr_cell: typing.Optional[Cell] = None
            for cell in tbl.cells:
                if not cell.text: 
                    continue
                cell_txt_low = cell.text.lower().strip()
               
                if "по данным продавца" in cell_txt_low: 
                    main_hdr_cell = cell 
                    break
               
                if "по данным" in cell_txt_low and any(v in cell_txt_low for v in sorted_seller_names):
                    main_hdr_cell = cell
                    break
            
            if not main_hdr_cell:
                self.logger.debug(f"Заголовок продавца не найден в табл. {tbl_idx + 1}.")
                continue
            
            self.logger.debug(f"Найден заголовок продавца: '{main_hdr_cell.text}' R{main_hdr_cell.row}C{main_hdr_cell.col}")

            debit_col, credit_col = self._find_debit_credit_columns_under_header(
                tbl.cells, main_hdr_cell.row, main_hdr_cell.col, main_hdr_cell.colspan)

            cols_ok = (debit_col != -1 and (main_hdr_cell.colspan < 2 or credit_col != -1))
            
            if not cols_ok:
                self.logger.warning(f"Не удалось идентифицировать Д/К колонки в табл. {tbl_idx + 1}.")
                continue
            
            self.logger.info(f"Колонки продавца: Дебет(C{debit_col})" + (f", Кредит(C{credit_col})" if credit_col!=-1 else ""))
            
            rows_map: typing.Dict[int, typing.Dict[int, str]] = {}
            for cell in tbl.cells:
                if cell.row not in rows_map: 
                    rows_map[cell.row] = {}
                
                rows_map[cell.row][cell.col] = cell.text.strip() if cell.text else ""

            data_start_row = main_hdr_cell.row + 2
            # добавил запоминание контекст "год", так как есть документы
            # где указан только месяц
            last_known_year_in_table: typing.Optional[int] = None

            for r_idx in sorted(rows_map.keys()):
                
                if r_idx < data_start_row: 
                    continue
                
                row_data = rows_map[r_idx]
                desc = " ".join(filter(None, (row_data.get(c_idx, "") for c_idx in range(debit_col)))).strip()


                date_val_str = None 
                if desc:
                    # Передаем last_known_year_in_table как контекст
                    date_info = extract_date_from_text(desc, self.logger, context_year=last_known_year_in_table)
                    if date_info:
                        date_val_str = date_info['formatted_str']
                        # Обновляем контекстный год, если извлеченная дата содержала год
                        if date_info.get('year') and date_info['year'] > 0: # Убедимся, что год валидный
                            last_known_year_in_table = date_info['year']
                
                debit_val = format_currency_value(row_data.get(debit_col, ""))
                credit_val = format_currency_value(row_data.get(credit_col, "")) if credit_col != -1 else ""
                
                transactions_data.append({
                    "table_idx": tbl_idx, 
                    "row_idx": r_idx, 
                    "record": desc,
                    "date": date_val_str, 
                    "debit": debit_val, 
                    "credit": credit_val
                })
                self.logger.info(f"  Т{tbl_idx}R{r_idx}: Оп='{desc}', Дата={date_val_str or None}, Д={debit_val}, К={credit_val}")
        return transactions_data
    
    def extract_for_buyer(self, buyer_info: dict) -> list[dict]:

        transactions_data = []
        buyer_names = set()
        
        if sr := buyer_info.get('str_repr'):
            sr_low = sr.lower()
            buyer_names.add(sr_low)
            buyer_names.add(sr_low.split(',')[0].strip())
            buyer_names.update(m.strip() for m in re.findall(r'\(([^,)]+)', sr_low) if m.strip() and len(m.strip()) > 1)
        buyer_names.update(cn.lower() for cn in buyer_info.get('canonical_names', []))
        
        if raw_txt := buyer_info.get('text'):
            raw_low = raw_txt.lower()
            core_raw = raw_low.split(',')[0].strip()
            buyer_names.add(core_raw)
            if qm := re.search(r'["«“]([^"»”]+)["»”]', core_raw): 
                buyer_names.add(qm.group(1).strip())
            buyer_names.update(m.strip() for m in re.findall(r'\(([^,)]+)', raw_low) if m.strip() and len(m.strip()) > 1)
        
        sorted_buyer_names = sorted([name for name in buyer_names if name], key=len, reverse=True)
        self.logger.debug(f"Варианты имени покупателя для поиска: {sorted_buyer_names}")

        for tbl_idx, tbl in enumerate(self.doc.get_tables()):
            self.logger.info(f"Анализ таблицы {tbl_idx + 1} для акта сверки покупателя.")
            main_hdr_cell: typing.Optional[Cell] = None
            for cell in tbl.cells:
                if not cell.text: 
                    continue
                cell_txt_low = cell.text.lower().strip()
               
                if "по данным покупателя" in cell_txt_low: 
                    main_hdr_cell = cell 
                    break
               
                if "по данным" in cell_txt_low and any(v in cell_txt_low for v in sorted_buyer_names):
                    main_hdr_cell = cell
                    break
            
            if not main_hdr_cell:
                self.logger.debug(f"Заголовок плкупателя не найден в табл. {tbl_idx + 1}.")
                continue
            
            self.logger.debug(f"Найден заголовок плкупателя: '{main_hdr_cell.text}' R{main_hdr_cell.row}C{main_hdr_cell.col}")

            debit_col, credit_col = self._find_debit_credit_columns_under_header(
                tbl.cells, main_hdr_cell.row, main_hdr_cell.col, main_hdr_cell.colspan)

            cols_ok = (debit_col != -1 and (main_hdr_cell.colspan < 2 or credit_col != -1))
            
            if not cols_ok:
                self.logger.warning(f"Не удалось идентифицировать Д/К колонки в табл. {tbl_idx + 1}.")
                continue
            
            self.logger.info(f"Колонки покупателя: Дебет(C{debit_col})" + (f", Кредит(C{credit_col})" if credit_col!=-1 else ""))
            transactions_data.append({
                "table" :  tbl_idx,
                "col_debit": debit_col,
                "col_credit": credit_col,
            })
        return transactions_data