# SDK Pullenti Lingvo, version 4.29, march 2025. Copyright (c) 2013-2025, Pullenti. All rights reserved.
# Non-Commercial Freeware and Commercial Software.
# This class is generated using the converter Unisharping (www.unisharping.ru) from Pullenti C# project.
# The latest version of the code is available on the site www.pullenti.ru

import typing
from pullenti.unisharp.Utils import Utils
from pullenti.unisharp.Misc import Stopwatch

from pullenti.ner.core.GetTextAttr import GetTextAttr
from pullenti.morph.MorphNumber import MorphNumber
from pullenti.ner.ReferentToken import ReferentToken
from pullenti.ner.keyword.KeywordReferent import KeywordReferent
from pullenti.ner.MetaToken import MetaToken
from pullenti.morph.MorphGender import MorphGender
from pullenti.ner.SourceOfAnalysis import SourceOfAnalysis
from pullenti.ner.core.NounPhraseParseAttr import NounPhraseParseAttr
from pullenti.ner.ProcessorService import ProcessorService
from pullenti.ner.core.MiscHelper import MiscHelper
from pullenti.ner.core.NounPhraseHelper import NounPhraseHelper
from pullenti.Sdk import Sdk
from pullenti.ner.ServerService import ServerService
from pullenti.ner.core.ReferentsEqualType import ReferentsEqualType
from pullenti.ner.keyword.KeywordAnalyzer import KeywordAnalyzer

class Program:
    
    @staticmethod
    def main(args : typing.List[str]) -> None:
        vers = ServerService.get_server_version(None)
        if (vers is not None): 
            print("Server version = {0}".format(vers), end="", flush=True)
            if (vers == ProcessorService.get_version()): 
                print(" (equal with current SDK)", flush=True)
            else: 
                print(" (but version of current SDK {0})!".format(ProcessorService.get_version()), flush=True)
            sres = ServerService.process_on_server(None, None, "В США прядь волос третьего президента Соединенных Штатов Томаса\nДжефферсона продали на аукционе в Техасе за 6,9 тысячи долларов,\nпередает Life. Локоны бывшего лидера США ушли с молотка почти\nчерез 190 лет после его смерти. Их покупатель пожелал остаться\nнеизвестным. Перед началом аукциона волосы Джефферсона оценивали\nв 3 тысячи долларов. В январе 2015 года прядь волос 16-го президента\nСША Авраама Линкольна продали за 25 тысяч долларов на аукционе в Далласе.", None)
            print("Server result: {0}".format(str(sres)), flush=True)
        sw = Stopwatch()
        # инициализация - необходимо проводить один раз до обработки текстов
        print("Initializing SDK Pullenti ver {0} ({1}) ... ".format(Sdk.get_version(), Sdk.get_version_date()), end="", flush=True)
        # инициализируются движок и все имеющиеся анализаторы
        Sdk.initialize_all()
        sw.stop()
        print("OK (by {0} ms), version {1}".format(sw.elapsedMilliseconds, ProcessorService.get_version()), flush=True)
        # посмотрим, какие анализаторы доступны
        for a in ProcessorService.get_analyzers(): 
            print("   {0} {1} \"{2}\"".format(("Specific analyzer" if a.is_specific else "Common analyzer"), a.name, a.caption), flush=True)
        # анализируемый текст
        txt = "Система разрабатывается с 2011 года российским программистом Михаилом Жуковым, проживающим в Москве на Красной площади в доме номер один на втором этаже. Конкурентов у него много: Abbyy, Yandex, ООО \"Russian Context Optimizer\" (RCO) и другие компании. Он планирует продать SDK за 1.120.000.001,99 (миллиард сто двадцать миллионов один рубль 99 копеек) рублей, без НДС."
        print("Text: {0}".format(txt), flush=True)
        # запускаем обработку на пустом процессоре (без анализаторов NER)
        are = ProcessorService.get_empty_processor().process(SourceOfAnalysis(txt), None, None)
        print("Noun groups: ", end="", flush=True)
        t = are.first_token
        # перебираем токены
        first_pass3733 = True
        while True:
            if first_pass3733: first_pass3733 = False
            else: t = t.next0_
            if (not (t is not None)): break
            # выделяем именную группу с текущего токена
            npt = NounPhraseHelper.try_parse(t, NounPhraseParseAttr.NO, 0, None)
            # не получилось
            if (npt is None): 
                continue
            # получилось, выводим в нормализованном виде
            print("[{0}=>{1}] ".format(npt.get_source_text(), npt.get_normal_case_text(None, MorphNumber.SINGULAR, MorphGender.UNDEFINED, False)), end="", flush=True)
            # указатель на последний токен именной группы
            t = npt.end_token
        with ProcessorService.create_processor() as proc: 
            # анализируем текст
            ar = proc.process(SourceOfAnalysis(txt), None, None)
            # результирующие сущности
            print("\r\n==========================================\r\nEntities: ", flush=True)
            for e0_ in ar.entities: 
                print("{0}: {1}".format(e0_.type_name, str(e0_)), flush=True)
                for s in e0_.slots: 
                    print("   {0}: {1}".format(s.type_name, s.value), flush=True)
            # пример выделения именных групп
            print("\r\n==========================================\r\nNoun groups: ", flush=True)
            t = ar.first_token
            first_pass3734 = True
            while True:
                if first_pass3734: first_pass3734 = False
                else: t = t.next0_
                if (not (t is not None)): break
                # токены с сущностями игнорируем
                if (t.get_referent() is not None): 
                    continue
                # пробуем создать именную группу
                npt = NounPhraseHelper.try_parse(t, NounPhraseParseAttr.ADJECTIVECANBELAST, 0, None)
                # не получилось
                if (npt is None): 
                    continue
                print(npt, flush=True)
                # указатель перемещаем на последний токен группы
                t = npt.end_token
            # попробуем проанализировать через сервер (если он запущен, естественно)
            server_address = "http://localhost:1111"
            server_sdk_version = ServerService.get_server_version(server_address)
            if (server_sdk_version is None): 
                print("Server not exists on {0}, OK".format(server_address), flush=True)
            else: 
                print("Server SDK Version: {0}".format(server_sdk_version), flush=True)
                # желательно проверить тождественность версий, а то мало ли...
                if (server_sdk_version != ProcessorService.get_version()): 
                    print("Server version {0} not equals current SDK version {1}".format(server_sdk_version, ProcessorService.get_version()), flush=True)
                else: 
                    try: 
                        # по идее, должны получить абсолютно эквивалентный результат, что и в ar
                        ar2 = ServerService.process_on_server(server_address, proc, txt, None)
                        if (ar2 is None): 
                            print("Server execution ERROR! ", flush=True)
                        elif (len(ar2.entities) != len(ar.entities)): 
                            print("Entities on server = {0}, but on local = {1}".format(len(ar2.entities), len(ar.entities)), flush=True)
                        else: 
                            eq = True
                            i = 0
                            while i < len(ar2.entities): 
                                if (not ar2.entities[i].can_be_equals(ar.entities[i], ReferentsEqualType.WITHINONETEXT)): 
                                    print("Server entity '{0}' not equal local entity '{1}'".format(ar2.entities[i], ar.entities[i]), flush=True)
                                    eq = False
                                i += 1
                            if (eq): 
                                print("Process on server equals local process!", flush=True)
                    except Exception as ex: 
                        print("Process on server error: " + ex.__str__(), flush=True)
        with ProcessorService.create_specific_processor(KeywordAnalyzer.ANALYZER_NAME) as proc: 
            ar = proc.process(SourceOfAnalysis(txt), None, None)
            print("\r\n==========================================\r\nKeywords1: ", flush=True)
            for e0_ in ar.entities: 
                if (isinstance(e0_, KeywordReferent)): 
                    print(e0_, flush=True)
            print("\r\n==========================================\r\nKeywords2: ", flush=True)
            t = ar.first_token
            first_pass3735 = True
            while True:
                if first_pass3735: first_pass3735 = False
                else: t = t.next0_
                if (not (t is not None)): break
                if (isinstance(t, ReferentToken)): 
                    kw = Utils.asObjectOrNull(t.get_referent(), KeywordReferent)
                    if (kw is None): 
                        continue
                    kwstr = MiscHelper.get_text_value_of_meta_token(Utils.asObjectOrNull(t, ReferentToken), Utils.valToEnum((GetTextAttr.FIRSTNOUNGROUPTONOMINATIVESINGLE) | (GetTextAttr.KEEPREGISTER), GetTextAttr))
                    print("{0} = {1}".format(kwstr, kw), flush=True)
        print("Over!", flush=True)

if __name__ == "__main__":
    Program.main(None)
