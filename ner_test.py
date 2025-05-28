import logging
import typing

from pullenti.ner.AnalysisResult import AnalysisResult
from pullenti.ner.ExtOntology import ExtOntology
from pullenti.ner.date.DateRangeReferent import DateRangeReferent
from pullenti.ner.date.DateReferent import DateReferent
from pullenti.ner.money.MoneyReferent import MoneyReferent
from pullenti.ner.org.OrganizationReferent import OrganizationReferent
from pullenti.ner.SourceOfAnalysis import SourceOfAnalysis
from pullenti.ner.ProcessorService import ProcessorService
from pullenti.Sdk import Sdk

from src.PDFExtractor.scan_extractor import ScanExtractor
from src.PDFExtractor.base_extractor import Document
import tests.common_test
from src.initialize_api import logger_configure


class NER:

    def __init__(self):
        self.logger = logging.getLogger('app.' + __class__.__name__)
        self.org_ontos = self._configure_org_ontology()
        self._configure_org_termins()
        self.org_types_full_names_map = {
            'АО': 'акционерное общество',
            'ОАО': 'открытое акционерное общество',
            'ЗАО': 'закрытое акционерное общество', 
            'ООО': 'общество с ограниченной ответственностью',
            'ИП': 'индивидуальный предприниматель',
            'ПАО': 'публичное акционерное общество'
        }
        self.pullenti_formal_types = [ft.lower() for ft in self.org_types_full_names_map.values()]

        self.seller_key_words = [

            'продавец', 

            'с одной стороны',
            'между'
        ]
        self.buyer_key_words = [

            'покупатель',

            'с другой стороны',
        ]

    def _extract_raw_organizations(self, analysis_result: AnalysisResult, txt: str) -> list[dict]:
        """Извлекает первичный список организаций из результатов анализа Pullenti."""
        raw_orgs = []
        for ent in analysis_result.entities:
            if not isinstance(ent, OrganizationReferent):
                continue

            current_org_ref = ent
            current_org_str = str(ent) 
            is_linked_to_custom_ontology = False

            if ent.ontology_items:
                ontology_linked_referent = ent.ontology_items[0].referent
                if ontology_linked_referent and isinstance(ontology_linked_referent, OrganizationReferent):
                    current_org_ref = ontology_linked_referent
                    current_org_str = str(ontology_linked_referent)
                    self.logger.debug(f"Entity '{str(ent)}' linked to ontology item: '{current_org_str}'")
                    is_linked_to_custom_ontology = True
            
            self.logger.debug(f"Pullenti raw: {current_org_ref.type_name}: {current_org_str}")

            is_primary_legal_entity = False
            identified_type_short_form = None 

            for slot in current_org_ref.slots:
                if slot.type_name == OrganizationReferent.ATTR_TYPE and isinstance(slot.value, str):
                    slot_value_lower = slot.value.lower()
                    slot_value_upper = slot.value.upper()
                    if slot_value_lower in self.pullenti_formal_types:
                        is_primary_legal_entity = True
                        for short_type, long_type_val in self.org_types_full_names_map.items():
                            if long_type_val.lower() == slot_value_lower:
                                identified_type_short_form = short_type
                                break
                    elif self.org_types_full_names_map.get(slot_value_upper) in self.pullenti_formal_types:
                        is_primary_legal_entity = True
                        identified_type_short_form = slot_value_upper 
                if is_primary_legal_entity and identified_type_short_form: 
                    break
            
            if not is_primary_legal_entity:
                continue

            if not ent.occurrence:
                self.logger.warning(f"OrganizationReferent '{str(ent)}' не имеет информации о вхождении (occurrence).")
                continue
            
            occ = ent.occurrence[0]
            org_text_from_doc = txt[occ.begin_char:occ.end_char]

            current_canonical_names = []
            for slot in current_org_ref.slots:
                if slot.type_name == OrganizationReferent.ATTR_NAME and isinstance(slot.value, str):
                    current_canonical_names.append(slot.value.upper())
            
            if not current_canonical_names and current_org_str:
                base_name_from_str_repr = current_org_str.split(',')[0].strip().upper()
                if base_name_from_str_repr:
                    current_canonical_names.append(base_name_from_str_repr)
            
            current_canonical_names = sorted(list(set(cn for cn in current_canonical_names if cn)))

            raw_orgs.append({
                "ref": current_org_ref,
                "text": org_text_from_doc,
                "str_repr": current_org_str,
                "canonical_names": current_canonical_names,
                "begin_char": occ.begin_char,
                "end_char": occ.end_char,
                "window": txt[occ.end_char : min(occ.end_char + 70, len(txt))].lower(),
                "is_linked_to_custom_ontology": is_linked_to_custom_ontology,
                "role": None
            })
        return raw_orgs

    def _filter_subsumed_organizations(self, orgs_list: list[dict]) -> list[dict]:
        """Фильтрует организации, поглощенные другими (связанными с онтологией)."""
        final_orgs_list = []
        for i, org_i in enumerate(orgs_list):
            is_subsumed = False
            if not org_i['is_linked_to_custom_ontology']:
                for j, org_j in enumerate(orgs_list):
                    if i == j or not org_j['is_linked_to_custom_ontology']:
                        continue
                    
                    if org_i['canonical_names'] and org_j['canonical_names']:
                        for name_i_canon in org_i['canonical_names']:
                            for name_j_canon in org_j['canonical_names']:
                                if name_i_canon in name_j_canon and len(name_i_canon) < len(name_j_canon):
                                    self.logger.debug(
                                        f"Организация '{org_i['str_repr']}' (общая) поглощена '{org_j['str_repr']}' (из онтологии) "
                                        f"из-за совпадения канонических имен ('{name_i_canon}' в '{name_j_canon}')."
                                    )
                                    is_subsumed = True
                                    break
                            if is_subsumed:
                                break
                    if is_subsumed:
                        break
            
            if not is_subsumed:
                final_orgs_list.append(org_i)
        return final_orgs_list

    def _assign_roles_by_keywords_and_rusal_logic(self, orgs_list: list[dict]) -> None:
        """Присваивает роли на основе ключевых слов и специальной логики для РУСАЛ."""
        for org_info in orgs_list:
            window_text = org_info['window']

            if any(keyword in window_text for keyword in self.seller_key_words):
                org_info['role'] = 'продавец'
                self.logger.debug(f"Организация '{org_info['str_repr']}' -> 'продавец' (ключевые слова в окне: '{window_text}')")
            
            if org_info['role'] is None and any(keyword in window_text for keyword in self.buyer_key_words):
                org_info['role'] = 'покупатель'
                self.logger.debug(f"Организация '{org_info['str_repr']}' -> 'покупатель' (ключевые слова в окне: '{window_text}')")

            if org_info['role'] is None:
                is_rusal_related = False
                if any("РУСАЛ" in canon_name for canon_name in org_info.get('canonical_names', [])):
                    is_rusal_related = True
                
                if not is_rusal_related:
                    org_name_for_check = org_info['str_repr'].split(',')[0].strip().upper()
                    if "РУСАЛ" in org_name_for_check:
                        is_rusal_related = True
                
                if is_rusal_related:
                    org_info['role'] = 'покупатель'
                    self.logger.debug(f"Организация '{org_info['str_repr']}' -> 'покупатель' (содержит 'РУСАЛ').")

    def _assign_mutual_roles(self, orgs_list: list[dict]) -> None:
        """Присваивает взаимоисключающие роли, если в списке ровно две организации."""
        if len(orgs_list) == 2:
            org1, org2 = orgs_list[0], orgs_list[1]

            if org1['role'] == 'покупатель' and org2['role'] is None:
                org2['role'] = 'продавец'
                self.logger.debug(f"Организация '{org2['str_repr']}' -> 'продавец' (т.к. '{org1['str_repr']}' покупатель).")
            elif org2['role'] == 'покупатель' and org1['role'] is None:
                org1['role'] = 'продавец'
                self.logger.debug(f"Организация '{org1['str_repr']}' -> 'продавец' (т.к. '{org2['str_repr']}' покупатель).")
            elif org1['role'] == 'продавец' and org2['role'] is None:
                org2['role'] = 'покупатель'
                self.logger.debug(f"Организация '{org2['str_repr']}' -> 'покупатель' (т.к. '{org1['str_repr']}' продавец).")
            elif org2['role'] == 'продавец' and org1['role'] is None:
                org1['role'] = 'покупатель'
                self.logger.debug(f"Организация '{org1['str_repr']}' -> 'покупатель' (т.к. '{org2['str_repr']}' продавец).")

    def _log_final_organization_roles(self, orgs_list: list[dict]) -> None:
        """Логирует итоговый список организаций с их ролями."""
        if orgs_list:
            self.logger.info("Итоговые организации и их роли:")
            for org_info in orgs_list:
                role_value = org_info.get('role', 'Не определена')
                role_str = role_value if role_value is not None else 'Не определена'
                
                log_entry = f"{role_str.upper()} - {org_info['str_repr']}"
                if not org_info['is_linked_to_custom_ontology']:
                    log_entry += f" (Исходный текст: '{org_info['text']}')"
                
                self.logger.info(log_entry)
        else:
            self.logger.warning("Не найдено организаций для назначения ролей после всех этапов.")

    def find_organization(self, txt: str) -> list[dict]:
        self.logger.debug(f"Начало обработки текста для поиска организаций:\n{txt[:500]}...") # Логируем начало текста
        
        with ProcessorService.create_processor() as proc:
            analysis_result = proc.process(SourceOfAnalysis(txt), self.org_ontos)
        
        # 1. Извлечение первичного списка организаций
        extracted_orgs = self._extract_raw_organizations(analysis_result, txt)
        if not extracted_orgs:
            self.logger.info("Первичное извлечение не дало организаций.")
            return []

        # 2. Фильтрация поглощенных организаций
        filtered_orgs = self._filter_subsumed_organizations(extracted_orgs)
        if not filtered_orgs:
            self.logger.info("После фильтрации поглощенных организаций список пуст.")
            return []
        
        # 3. Присвоение ролей по ключевым словам и логике РУСАЛ
        self._assign_roles_by_keywords_and_rusal_logic(filtered_orgs)
        
        # 4. Присвоение взаимоисключающих ролей (если применимо)
        self._assign_mutual_roles(filtered_orgs)
        
        # 5. Логирование результатов
        self._log_final_organization_roles(filtered_orgs)
        
        return filtered_orgs
    
    def _configure_org_ontology(self) -> ExtOntology: # Переименовано в приватный
        org_ontos = ExtOntology()
        map_orgs = {
            'РУСАЛ НОВОКУЗНЕЦКИЙ АЛЮМИНИЕВЫЙ ЗАВОД': 'АО',
            'РУСАЛ АЧИНСКИЙ ГЛИНОЗЕМНЫЙ КОМБИНАТ': 'АО',
            'ОК РУСАЛ ТД': 'АО'
        }
        
        org_id_counter = 0
        for org_full_name, org_type in map_orgs.items():
            ontology_item_ref = OrganizationReferent()
            names_to_add_to_ontology = set()
            
            org_full_name_upper = org_full_name.upper()
            words = org_full_name_upper.split()

            names_to_add_to_ontology.add(org_full_name_upper)

            if len(words) > 1 and words[0] == "РУСАЛ":
                specific_part = " ".join(words[1:])
                names_to_add_to_ontology.add(specific_part)

            if len(words) > 1:
                current_prefix_parts = []
                for i in range(len(words) - 1):
                    current_prefix_parts.append(words[i])
                    prefix_variant = " ".join(current_prefix_parts)
                    if not (prefix_variant == "РУСАЛ" and words[0] == "РУСАЛ" and len(words) > 1):
                        names_to_add_to_ontology.add(prefix_variant)
            
            if org_full_name_upper == "РУСАЛ": # На случай, если "РУСАЛ" сам по себе в map_orgs
                 names_to_add_to_ontology.add("РУСАЛ")

            self.logger.debug(f"Алиасы для онтологии '{org_full_name_upper}': {names_to_add_to_ontology}")

            for name_variant in names_to_add_to_ontology:
                ontology_item_ref.add_slot(OrganizationReferent.ATTR_NAME, name_variant, clear_old_value=False)
            ontology_item_ref.add_slot(OrganizationReferent.ATTR_TYPE, org_type.upper(), clear_old_value=False)
            
            org_ontos.add_referent(f"org_{org_id_counter}", ontology_item_ref)
            org_id_counter += 1
        return org_ontos

    def _configure_org_termins(self) -> None: # Переименовано в приватный
        # Этот метод не используется в find_organization, но оставлен для полноты
        # Если он нужен для Pullenti, его нужно будет передавать в proc.process или инициализировать глобально
        termins_map = {
            'РУСАЛ': 'Русский Алюминий'
        }
        
        # Для использования TerminCollection с ProcessorService, их обычно регистрируют через Sdk.
        # Пример:
        # termin_collection = TerminCollection()
        # for key, value in termins_map.items():
        #     term = Termin(value)
        #     term.add_abridge(key)
        #     termin_collection.add(term)
        # KeywordAnalyzer.GLOBAL_ONTOLOGY.add_ext(termin_collection) 
        # Либо передавать в specific_analyzers_data при создании процессора, если это поддерживается.
        # В текущей реализации PullentiNetPthon, ExtOntology - основной способ кастомизации для NER.
        self.logger.debug(f"Конфигурация терминов: {termins_map} (в текущей реализации неактивно влияет на NER)")

    

class Program:

    @staticmethod
    def main(args: typing.List[str]) -> None:
        logger_configure()
        logger = logging.getLogger('app.program')
        Sdk.initialize_all()
        extractor_scan = ScanExtractor()
        pdf_bytes = tests.common_test.get_pdf_scan()
        doc_structure: Document = extractor_scan.extract(pdf_bytes)
        

        full_text = []
        for page in doc_structure.pages:
            for para in page.paragraphs:
                if not para.text:
                    continue
                full_text.append(para.text)
        
        concatenated_text = "\n".join(full_text)
        morph = NER()
        identified_organizations = morph.find_organization(concatenated_text)

        


if __name__ == "__main__":
    Program.main([])