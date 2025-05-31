
import logging

from pullenti.ner.AnalysisResult import AnalysisResult
from pullenti.ner.ExtOntology import ExtOntology
from pullenti.ner.ProcessorService import ProcessorService
from pullenti.ner.SourceOfAnalysis import SourceOfAnalysis
from pullenti.ner.org.OrganizationAnalyzer import OrganizationAnalyzer
from pullenti.ner.org.OrganizationReferent import OrganizationReferent


class OrganizationProcessor:
    """
    Отвечает за извлечение, фильтрацию и назначение ролей организациям из текста.
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.org_types_full_names_map = {
            'АО': 'акционерное общество', 'ОАО': 'открытое акционерное общество',
            'ЗАО': 'закрытое акционерное общество', 'ООО': 'общество с ограниченной ответственностью',
            'ИП': 'индивидуальный предприниматель', 'ПАО': 'публичное акционерное общество'
        }
        self.pullenti_formal_types = [ft.lower() for ft in self.org_types_full_names_map.values()]
        self.seller_key_words = ['продавец', 'с одной стороны', 'между']
        self.buyer_key_words = ['покупатель', 'с другой стороны']
        self.org_ontos = self._configure_org_ontology()

    def _configure_org_ontology(self) -> ExtOntology:
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
                names_to_add_to_ontology.add(" ".join(words[1:]))
            if len(words) > 1:
                current_prefix_parts = []
                for i in range(len(words) - 1):
                    current_prefix_parts.append(words[i])
                    prefix_variant = " ".join(current_prefix_parts)
                    if not (prefix_variant == "РУСАЛ" and words[0] == "РУСАЛ" and len(words) > 1):
                        names_to_add_to_ontology.add(prefix_variant)
            if org_full_name_upper == "РУСАЛ": 
                names_to_add_to_ontology.add("РУСАЛ")

            self.logger.debug(f"Алиасы для онтологии '{org_full_name_upper}': {names_to_add_to_ontology}")
            for name_variant in names_to_add_to_ontology:
                ontology_item_ref.add_slot(OrganizationReferent.ATTR_NAME, name_variant, False)
            ontology_item_ref.add_slot(OrganizationReferent.ATTR_TYPE, org_type.upper(), False)
            org_ontos.add_referent(f"org_{org_id_counter}", ontology_item_ref)
            org_id_counter += 1
        return org_ontos

    def _extract_raw_organizations(self, analysis_result: AnalysisResult, txt: str) -> list[dict]:
        raw_orgs = []
        for ent in analysis_result.entities:
            
            if not isinstance(ent, OrganizationReferent): 
                continue
            
            current_org_ref, current_org_str = ent, str(ent)
            is_linked_to_custom_ontology = False
            if ent.ontology_items and isinstance(ent.ontology_items[0].referent, OrganizationReferent):
                current_org_ref = ent.ontology_items[0].referent
                current_org_str = str(current_org_ref)
                self.logger.debug(f"Entity '{str(ent)}' linked to ontology item: '{current_org_str}'")
                is_linked_to_custom_ontology = True
            
            self.logger.debug(f"Pullenti raw: {current_org_ref.type_name}: {current_org_str}")
            is_primary_legal_entity = False
            for slot in current_org_ref.slots:
                if slot.type_name == OrganizationReferent.ATTR_TYPE and isinstance(slot.value, str):
                    slot_val_low = slot.value.lower()
                    if slot_val_low in self.pullenti_formal_types or \
                       self.org_types_full_names_map.get(slot.value.upper(), "").lower() in self.pullenti_formal_types:
                        is_primary_legal_entity = True 
                        break
            if not is_primary_legal_entity: 
                continue
            if not ent.occurrence:
                self.logger.warning(f"Org '{str(ent)}' no occurrence.") 
                continue
            
            occ = ent.occurrence[0]
            org_text_from_doc = txt[occ.begin_char:occ.end_char]
            can_names = [s.value.upper() for s in current_org_ref.slots if 
                         s.type_name == OrganizationReferent.ATTR_NAME and isinstance(s.value, str)]
            
            if not can_names and current_org_str:
                base_name = current_org_str.split(',')[0].strip().upper()
                if base_name: 
                    can_names.append(base_name)
            
            raw_orgs.append({
                "text": org_text_from_doc, "str_repr": current_org_str,
                "canonical_names": sorted(list(set(cn for cn in can_names if cn))),
                "window": txt[occ.end_char : min(occ.end_char + 70, len(txt))].lower(),
                "is_linked_to_custom_ontology": is_linked_to_custom_ontology, "role": None
            })
        return raw_orgs

    def _filter_subsumed_organizations(self, orgs_list: list[dict]) -> list[dict]:
        final_orgs = []
        for i, org_i in enumerate(orgs_list):
            is_subsumed = False
            if not org_i['is_linked_to_custom_ontology']:
                for j, org_j in enumerate(orgs_list):
                    if i == j or not org_j['is_linked_to_custom_ontology']: 
                        continue
                    if any(name_i in name_j and len(name_i) < len(name_j)
                           for name_i in org_i.get('canonical_names', [])
                           for name_j in org_j.get('canonical_names', [])):
                        self.logger.debug(f"'{org_i['str_repr']}' subsumed by '{org_j['str_repr']}' (ontology)")
                        is_subsumed = True
                        break
            if not is_subsumed: 
                final_orgs.append(org_i)
        return final_orgs

    def _assign_roles_by_keywords_and_rusal_logic(self, orgs_list: list[dict]) -> None:
        for org in orgs_list:
            win_text = org['window']
            if any(kw in win_text for kw in self.seller_key_words):
                org['role'] = 'продавец'
            elif any(kw in win_text for kw in self.buyer_key_words): 
                org['role'] = 'покупатель'
            
            if org['role'] is None and \
               (any("РУСАЛ" in cn for cn in org.get('canonical_names', [])) or \
                "РУСАЛ" in org['str_repr'].split(',')[0].strip().upper()):
                org['role'] = 'покупатель'
                self.logger.debug(f"'{org['str_repr']}' -> 'покупатель' (РУСАЛ logic).")

    def _assign_mutual_roles(self, orgs_list: list[dict]) -> None:
        if len(orgs_list) == 2:
            o1, o2 = orgs_list[0], orgs_list[1]
            if o1['role'] == 'покупатель' and o2['role'] is None: 
                o2['role'] = 'продавец'
            elif o2['role'] == 'покупатель' and o1['role'] is None: 
                o1['role'] = 'продавец'
            elif o1['role'] == 'продавец' and o2['role'] is None: 
                o2['role'] = 'покупатель'
            elif o2['role'] == 'продавец' and o1['role'] is None: 
                o1['role'] = 'покупатель'

    def _log_final_organization_roles(self, orgs_list: list[dict]) -> None:
        
        if not orgs_list: 
            self.logger.warning("Организации для назначения ролей не найдены.")
            return
        self.logger.info("Итоговые организации и их роли:")
        
        for org in orgs_list:
            role = org.get('role', 'Не определена')
            log_entry = f"{role.upper()} - {org['str_repr']}"
            if not org['is_linked_to_custom_ontology']: 
                log_entry += f" (Исходный: '{org['text']}')"
            self.logger.info(log_entry)

    def process_text(self, text: str) -> list[dict]:
        self.logger.debug(f"Поиск организаций в тексте (начало): {text[:200]}...")
        with ProcessorService.create_specific_processor(OrganizationAnalyzer.ANALYZER_NAME) as proc:
            res = proc.process(SourceOfAnalysis(text), self.org_ontos)
        
        orgs = self._extract_raw_organizations(res, text)
        if not orgs: 
            self.logger.info("Первичное извлечение не дало организаций.")
            return []
        
        orgs = self._filter_subsumed_organizations(orgs)
        if not orgs: 
            self.logger.info("После фильтрации поглощенных организаций список пуст.")
            return []
        
        self._assign_roles_by_keywords_and_rusal_logic(orgs)
        self._assign_mutual_roles(orgs)
        self._log_final_organization_roles(orgs)
        return orgs