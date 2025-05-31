
import logging
import unittest

import sys # Добавить этот импорт
import os # Добавить этот импорт

# Добавить путь к корневой папке проекта
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import common_test

from pullenti.Sdk import Sdk
from src.initialize_api import logger_configure
from src.NER.ner_service import NERService
from src.PDFExtractor.scan_extractor import ScanExtractor
from src.PDFExtractor.base_extractor import Document

class Test_TestNer(unittest.TestCase):

    def setUp(self):
        logger_configure()
        self.logger = logging.getLogger('test')
        self.scan_extractor = ScanExtractor()
        Sdk.initialize_all()

    
    def test_ner_seller_analization(self):
        try:
            pdf_bytes = common_test.get_pdf_scan() # Используем ваш метод для загрузки PDF
            if not pdf_bytes:
                self.logger.error("Не удалось загрузить PDF байты.")
                return
            
            scan_extractor = ScanExtractor() # Предполагаем, что это ваш экстрактор
            doc_structure: Document = scan_extractor.extract(pdf_bytes)
            
            
            ner_service = NERService(doc_structure)
            organizations = ner_service.find_document_organizations()

            seller = next((org for org in organizations if org.get('role') == 'продавец'), None)
            
            if seller:
                self.logger.info(f"Продавец определен: {seller.get('str_repr', 'N/A')}")
                reconciliation_details = ner_service.extract_seller_reconciliation_details(seller)
                if reconciliation_details:
                    self.logger.info(f"Извлечено {len(reconciliation_details)} транзакций для акта сверки продавца.")
                else:
                    self.logger.info("Транзакции для акта сверки продавца не найдены.")
            else:
                self.logger.warning("Продавец не определен. Анализ акта сверки не выполнен.")

        except Exception as e:
            self.logger.exception(f"Критическая ошибка в Program.main: {e}")
    
    def test_ner_buyer_analization(self):
        try:
            pdf_bytes = common_test.get_pdf_scan() # Используем ваш метод для загрузки PDF
            if not pdf_bytes:
                self.logger.error("Не удалось загрузить PDF байты.")
                return
            
            scan_extractor = ScanExtractor() # Предполагаем, что это ваш экстрактор
            doc_structure: Document = scan_extractor.extract(pdf_bytes)
            
            
            ner_service = NERService(doc_structure)
            organizations = ner_service.find_document_organizations()

            seller = next((org for org in organizations if org.get('role') == 'покупатель'), None)
            
            if seller:
                self.logger.info(f"Покупатель определен: {seller.get('str_repr', 'N/A')}")
                reconciliation_details = ner_service.extract_buyer_reconciliation_details(seller)
                if reconciliation_details:
                    self.logger.info(f"Извлечено {len(reconciliation_details)} транзакций для акта сверки покупателя.")
                else:
                    self.logger.info("Транзакции для акта сверки покупателя не найдены.")
            else:
                self.logger.warning("Покупатель не определен. Анализ акта сверки не выполнен.")

        except Exception as e:
            self.logger.exception(f"Критическая ошибка в Program.main: {e}")

if __name__ == '__main__':
    unittest.main()