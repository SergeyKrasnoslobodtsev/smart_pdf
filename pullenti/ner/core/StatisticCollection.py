﻿# SDK Pullenti Lingvo, version 4.29, march 2025. Copyright (c) 2013-2025, Pullenti. All rights reserved.
# Non-Commercial Freeware and Commercial Software.
# This class is generated using the converter Unisharping (www.unisharping.ru) from Pullenti C# project.
# The latest version of the code is available on the site www.pullenti.ru

from pullenti.unisharp.Utils import Utils
from pullenti.unisharp.Misc import RefOutArgWrapper

from pullenti.morph.MorphWordForm import MorphWordForm
from pullenti.ner.core.StatisticBigrammInfo import StatisticBigrammInfo
from pullenti.morph.MorphGender import MorphGender
from pullenti.ner.core.StatisticWordInfo import StatisticWordInfo
from pullenti.ner.TextToken import TextToken
from pullenti.ner.core.MiscHelper import MiscHelper

class StatisticCollection:
    """ Статистическая информация о словоформах и их биграммах в тексте - поле AnalysisKit.Statistic.
    Статистика
    """
    
    def __init__(self) -> None:
        self.__m_items = dict()
        self.__m_bigramms = dict()
        self.__m_bigramms_rev = dict()
        self.__m_initials = dict()
        self.__m_initials_rev = dict()
    
    def _prepare(self, first : 'Token') -> None:
        prev = None
        prevt = None
        t = first
        first_pass3858 = True
        while True:
            if first_pass3858: first_pass3858 = False
            else: t = t.next0_
            if (not (t is not None)): break
            if (t.is_hiphen): 
                continue
            it = None
            if (((isinstance(t, TextToken)) and t.chars.is_letter and t.length_char > 1) and not t.chars.is_all_lower): 
                it = self.__add_token(Utils.asObjectOrNull(t, TextToken))
            elif ((((isinstance(t, TextToken)) and t.length_char == 1 and t.chars.is_all_upper) and t.next0_ is not None and t.next0_.is_char('.')) and not t.is_whitespace_after): 
                it = self.__add_token(Utils.asObjectOrNull(t, TextToken))
                t = t.next0_
            if (prev is not None and it is not None): 
                self.__add_bigramm(prev, it)
                if (prevt.chars.equals(t.chars)): 
                    prev.add_after(it)
                    it.add_before(prev)
            prev = it
            prevt = t
        t = first
        while t is not None: 
            if (t.chars.is_letter and (isinstance(t, TextToken))): 
                it = self.__find_item(Utils.asObjectOrNull(t, TextToken), False)
                if (it is not None): 
                    if (t.chars.is_all_lower): 
                        it.lower_count += 1
                    elif (t.chars.is_all_upper): 
                        it.upper_count += 1
                    elif (t.chars.is_capital_upper): 
                        it.capital_count += 1
            t = t.next0_
    
    def __add_token(self, tt : 'TextToken') -> 'StatisticWordInfo':
        vars0_ = list()
        vars0_.append(tt.term)
        s = MiscHelper.get_absolute_normal_value(tt.term, False)
        if (s is not None and not s in vars0_): 
            vars0_.append(s)
        for wff in tt.morph.items: 
            wf = Utils.asObjectOrNull(wff, MorphWordForm)
            if (wf is None): 
                continue
            if (wf.normal_case is not None and not wf.normal_case in vars0_): 
                vars0_.append(wf.normal_case)
            if (wf.normal_full is not None and not wf.normal_full in vars0_): 
                vars0_.append(wf.normal_full)
        res = None
        for v in vars0_: 
            wrapres710 = RefOutArgWrapper(None)
            inoutres711 = Utils.tryGetValue(self.__m_items, v, wrapres710)
            res = wrapres710.value
            if (inoutres711): 
                break
        if (res is None): 
            res = StatisticWordInfo._new712(tt.lemma)
        for v in vars0_: 
            if (not v in self.__m_items): 
                self.__m_items[v] = res
        res.total_count += 1
        if ((isinstance(tt.next0_, TextToken)) and tt.next0_.chars.is_all_lower): 
            if (tt.next0_.chars.is_cyrillic_letter and tt.next0_.get_morph_class_in_dictionary().is_verb): 
                g = tt.next0_.morph.gender
                if (g == MorphGender.FEMINIE): 
                    res.female_verbs_after_count += 1
                elif (((g) & (MorphGender.MASCULINE)) != (MorphGender.UNDEFINED)): 
                    res.male_verbs_after_count += 1
        if (tt.previous is not None): 
            if ((isinstance(tt.previous, TextToken)) and tt.previous.chars.is_letter and not tt.previous.chars.is_all_lower): 
                pass
            else: 
                res.not_capital_before_count += 1
        return res
    
    def __find_item(self, tt : 'TextToken', do_absolute : bool=True) -> 'StatisticWordInfo':
        if (tt is None): 
            return None
        res = None
        wrapres719 = RefOutArgWrapper(None)
        inoutres720 = Utils.tryGetValue(self.__m_items, tt.term, wrapres719)
        res = wrapres719.value
        if (inoutres720): 
            return res
        if (do_absolute): 
            s = MiscHelper.get_absolute_normal_value(tt.term, False)
            if (s is not None): 
                wrapres713 = RefOutArgWrapper(None)
                inoutres714 = Utils.tryGetValue(self.__m_items, s, wrapres713)
                res = wrapres713.value
                if (inoutres714): 
                    return res
        for wff in tt.morph.items: 
            wf = Utils.asObjectOrNull(wff, MorphWordForm)
            if (wf is None): 
                continue
            wrapres717 = RefOutArgWrapper(None)
            inoutres718 = Utils.tryGetValue(self.__m_items, Utils.ifNotNull(wf.normal_case, ""), wrapres717)
            res = wrapres717.value
            if (inoutres718): 
                return res
            wrapres715 = RefOutArgWrapper(None)
            inoutres716 = Utils.tryGetValue(self.__m_items, wf.normal_full, wrapres715)
            res = wrapres715.value
            if (wf.normal_full is not None and inoutres716): 
                return res
        return None
    
    def __add_bigramm(self, b1 : 'StatisticWordInfo', b2 : 'StatisticWordInfo') -> None:
        di = None
        wrapdi723 = RefOutArgWrapper(None)
        inoutres724 = Utils.tryGetValue(self.__m_bigramms, b1.normal, wrapdi723)
        di = wrapdi723.value
        if (not inoutres724): 
            di = dict()
            self.__m_bigramms[b1.normal] = di
        if (b2.normal in di): 
            di[b2.normal] += 1
        else: 
            di[b2.normal] = 1
        wrapdi721 = RefOutArgWrapper(None)
        inoutres722 = Utils.tryGetValue(self.__m_bigramms_rev, b2.normal, wrapdi721)
        di = wrapdi721.value
        if (not inoutres722): 
            di = dict()
            self.__m_bigramms_rev[b2.normal] = di
        if (b1.normal in di): 
            di[b1.normal] += 1
        else: 
            di[b1.normal] = 1
    
    def get_bigramm_info(self, t1 : 'Token', t2 : 'Token') -> 'StatisticBigrammInfo':
        """ Получить статистическую информацию о биграмме токенов
        
        Args:
            t1(Token): первый токен биграммы
            t2(Token): второй токен биграммы
        
        Returns:
            StatisticBigrammInfo: информация о биграмме по всему тексту
        
        """
        si1 = self.__find_item(Utils.asObjectOrNull(t1, TextToken), True)
        si2 = self.__find_item(Utils.asObjectOrNull(t2, TextToken), True)
        if (si1 is None or si2 is None): 
            return None
        return self.__get_bigrams_info(si1, si2)
    
    def __get_bigrams_info(self, si1 : 'StatisticWordInfo', si2 : 'StatisticWordInfo') -> 'StatisticBigrammInfo':
        res = StatisticBigrammInfo._new725(si1.total_count, si2.total_count)
        di12 = None
        wrapdi12727 = RefOutArgWrapper(None)
        Utils.tryGetValue(self.__m_bigramms, si1.normal, wrapdi12727)
        di12 = wrapdi12727.value
        di21 = None
        wrapdi21726 = RefOutArgWrapper(None)
        Utils.tryGetValue(self.__m_bigramms_rev, si2.normal, wrapdi21726)
        di21 = wrapdi21726.value
        if (di12 is not None): 
            if (not si2.normal in di12): 
                res.first_has_other_second = True
            else: 
                res.pair_count = di12[si2.normal]
                if (len(di12) > 1): 
                    res.first_has_other_second = True
        if (di21 is not None): 
            if (not si1.normal in di21): 
                res.second_has_other_first = True
            elif (not si1.normal in di21): 
                res.second_has_other_first = True
            elif (len(di21) > 1): 
                res.second_has_other_first = True
        return res
    
    def get_initial_info(self, ini : str, sur : 'Token') -> 'StatisticBigrammInfo':
        if (Utils.isNullOrEmpty(ini)): 
            return None
        si2 = self.__find_item(Utils.asObjectOrNull(sur, TextToken), True)
        if (si2 is None): 
            return None
        si1 = None
        wrapsi1728 = RefOutArgWrapper(None)
        inoutres729 = Utils.tryGetValue(self.__m_items, ini[0:0+1], wrapsi1728)
        si1 = wrapsi1728.value
        if (not inoutres729): 
            return None
        if (si1 is None): 
            return None
        return self.__get_bigrams_info(si1, si2)
    
    def get_word_info(self, t : 'Token') -> 'StatisticWordInfo':
        """ Получить информацию о словоформе токена
        
        Args:
            t(Token): токен
        
        Returns:
            StatisticWordInfo: статистическая информация по тексту
        
        """
        tt = Utils.asObjectOrNull(t, TextToken)
        if (tt is None): 
            return None
        return self.__find_item(tt, True)