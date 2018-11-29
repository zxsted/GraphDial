# -*- coding: utf-8 -*-
# Author: Tiancheng Zhao
# Date: 9/13/17

import numpy as np
from simdial.agent.core import SystemAct, UserAct, BaseUsrSlot
from simdial.agent import core
import json
import copy


class AbstractNlg(object):
    """
    Abstract class of NLG
    """

    def __init__(self, domain, complexity):
        self.domain = domain
        self.complexity = complexity

    def generate_sent(self, actions, **kwargs):
        """
        Map a list of actions to a string.

        :param actions: a list of actions
        :return: uttearnces in string
        """
        raise NotImplementedError("Generate sent is required for NLG")

    def sample(self, examples):
        return np.random.choice(examples)


class SysCommonNlg(object):
    templates = {SystemAct.GREET: ["Hello.", "Hi.", "Greetings.", "How are you doing?"],
                 SystemAct.ASK_REPEAT: ["Can you please repeat that?", "What did you say?"],
                 SystemAct.ASK_REPHRASE: ["Can you please rephrase that?", "Can you say it in another way?"],
                 SystemAct.GOODBYE: ["Goodbye.", "See you next time."],
                 SystemAct.CLARIFY: ["I didn't catch you."],
                 SystemAct.REQUEST+core.BaseUsrSlot.NEED: ["What can I do for you?",
                                                           "What do you need?",
                                                           "How can I help?"],
                 SystemAct.REQUEST+core.BaseUsrSlot.HAPPY: ["What else can I do?",
                                                            "Are you happy about my answer?",
                                                            "Anything else?"],
                 SystemAct.EXPLICIT_CONFIRM+"dont_care": ["Okay, you dont_care, do you?",
                                                          "You dont_care, right?"],
                 SystemAct.IMPLICIT_CONFIRM+"dont_care": ["Okay, you dont_care.",
                                                          "Alright, dont_care."]}

class SysNlg(AbstractNlg):
    """
    NLG class to generate utterances for the system side.
    """

    def generate_sent(self, actions, domain=None, templates=SysCommonNlg.templates):
        """
         Map a list of system actions to a string.
         将一个 system action 列表映射成 string

        :param actions: a list of actions        系统动作列表
        :param templates: a common NLG template that uses the default one if not given     # NLG 模板
        :return: uttearnces in string            # 返回 系统语句
        """
        str_actions = []
        lexicalized_actions = []
        for a in actions:
            a_copy = copy.deepcopy(a)
            if a.act == SystemAct.GREET:
                if domain:
                    str_actions.append(domain.greet)   # 输出当前系统的 问候语句
                else:
                    str_actions.append(self.sample(templates[a.act]))   # 如果没有指定则输出通用模板的问候语句

            elif a.act == SystemAct.QUERY:              # 如果系统动作是 query(访问数据库的查询语句)
                usr_constrains = a.parameters[0]        # 系统追踪的 用户约束值
                sys_goals = a.parameters[1]             # 系统目标槽位

                # create string list for KB_SEARCH
                search_dict = {}
                for k, v in usr_constrains:
                    slot = self.domain.get_usr_slot(k)      # 取出 对应的槽位对象
                    if v is None:
                        search_dict[k] = 'dont_care'        # 如果 槽位value时 空的则表示这个值不重要，可以忽略
                    else:
                        search_dict[k] = slot.vocabulary[v] # value是一个值的数值索引，重slot的词表中取出对应的真实值

                a_copy.parameters[0] = search_dict
                a_copy.parameters[1] = sys_goals
                # 这时以json的格式返回 查询语句，和查询目标
                str_actions.append(json.dumps({"QUERY": search_dict,
                                               "GOALS": sys_goals}))

            elif a.act == SystemAct.INFORM:      # 当前系统的动作是 inform
                sys_goals = a.parameters[1]      # 取出 goal的槽位值

                # create string list for RET + Informs
                informs = []
                sys_goal_dict = {}
                for k, (v, e_v) in sys_goals.items():      # 取出 槽位名称 槽位 追踪值索引 期望值索引
                    slot = self.domain.get_sys_slot(k)     # 取出槽位对应的 对象
                    sys_goal_dict[k] = slot.vocabulary[v]  # 取出 值索引对应的真实值

                    #如果 期望值和追踪值相同， 那么是 user say 前缀添加 yes
                    if e_v is not None:
                        prefix = "Yes, " if v == e_v else "No, "
                    else:
                        prefix = ""
                    # 前缀 + slot 采样的模板 + slot 真实值
                    informs.append(prefix + slot.sample_inform()
                                   % slot.vocabulary[v])
                # 包村sys——gaol dict
                a_copy['parameters'] = [sys_goal_dict]
                # 拼接 inform 列表
                str_actions.append(" ".join(informs))

            elif a.act == SystemAct.REQUEST:          #如果当前系统的动作是request
                slot_type, _ = a.parameters[0]        #取出slot 的name
                if slot_type in [core.BaseUsrSlot.NEED, core.BaseUsrSlot.HAPPY]:    #如果槽位时need 或者 happy 从模板中采样出user say
                    str_actions.append(self.sample(templates[SystemAct.REQUEST+slot_type]))
                else:
                    target_slot = self.domain.get_usr_slot(slot_type)       # 取出对应的 target slot 对象
                    if target_slot is None:
                        raise ValueError("none slot %s" % slot_type)
                    str_actions.append(target_slot.sample_request())        # 从slot 对象中随机采样出一个erquest

            elif a.act == SystemAct.EXPLICIT_CONFIRM:                   # 如果系统的动作是不确定澄清
                slot_type, slot_val = a.parameters[0]
                if slot_val is None:      # 如果 slot val是none ,那么从模板中采样的是 这个槽位是否可以忽略
                    str_actions.append(self.sample(templates[SystemAct.EXPLICIT_CONFIRM+"dont_care"]))
                    a_copy.parameters[0] = (slot_type, "dont_care")
                else:
                    # 否则询问用户是不是这个曹值
                    slot = self.domain.get_usr_slot(slot_type)
                    str_actions.append("Do you mean %s?"
                                       % slot.vocabulary[slot_val])
                    a_copy.parameters[0] = (slot_type, slot.vocabulary[slot_val])

            elif a.act == SystemAct.IMPLICIT_CONFIRM:      # 系统是显式澄清
                slot_type, slot_val = a.parameters[0]
                if slot_val is None:   # 同上
                    str_actions.append(self.sample(templates[SystemAct.IMPLICIT_CONFIRM+"dont_care"]))
                    a_copy.parameters[0] = (slot_type, "dont_care")
                else:    #同上 确定性反问
                    slot = self.domain.get_usr_slot(slot_type)
                    str_actions.append("I believe you said %s."
                                       % slot.vocabulary[slot_val])
                    a_copy.parameters[0] = (slot_type, slot.vocabulary[slot_val])

            elif a.act in templates.keys():    # 否则如果在模板中，就随机选一个回复
                str_actions.append(self.sample(templates[a.act]))

            else:
                raise ValueError("Unknown dialog act %s" % a.act)

            lexicalized_actions.append(a_copy)     # 用于json 展示

        return " ".join(str_actions), lexicalized_actions


class UserNlg(AbstractNlg):
    """
    NLG class to generate utterances for the user side.
    """

    def generate_sent(self, actions):
        """
         Map a list of user actions to a string.

        :param actions: a list of actions
        :return: uttearnces in string
        """
        str_actions = []
        for a in actions:
            if a.act == UserAct.KB_RETURN:     # 如果是数据库查询结果， 就将查询到的内容返以json'的形式返回
                sys_goals = a.parameters[1]
                sys_goal_dict = {}
                for k, v in sys_goals.items():
                    slot = self.domain.get_sys_slot(k)
                    sys_goal_dict[k] = slot.vocabulary[v]

                str_actions.append(json.dumps({"RET": sys_goal_dict}))
            elif a.act == UserAct.GREET:     # 问候就返回问候
                str_actions.append(self.sample(["Hi.", "Hello robot.", "What's up?"]))

            elif a.act == UserAct.GOODBYE:  # 结束就返回结束语句
                str_actions.append(self.sample(["That's all.", "Thank you.", "See you."]))

            elif a.act == UserAct.REQUEST:       # 如果是request， 就从 对应的slot中采样出 对应的request 语句
                slot_type, _ = a.parameters[0]
                target_slot = self.domain.get_sys_slot(slot_type)
                str_actions.append(target_slot.sample_request())

            elif a.act == UserAct.INFORM:          # 如果是 inform 动作
                has_self_correct = a.parameters[-1][0] == BaseUsrSlot.SELF_CORRECT    # 判断是不是自己错误
                slot_type, slot_value = a.parameters[0]
                target_slot = self.domain.get_usr_slot(slot_type)

                def get_inform_utt(val):
                    if val is None:
                        return self.sample(["Anything is fine.", "I don't care.", "Whatever is good."])
                    else:
                        return target_slot.sample_inform() % target_slot.vocabulary[val]

                if has_self_correct:     # 如果是自己错误
                    wrong_value = target_slot.sample_different(slot_value)    # 随机采样一个其他的曹值
                    wrong_utt = get_inform_utt(wrong_value)                   # 使用错误值生成inform 语句
                    correct_utt = get_inform_utt(slot_value)                  # 使用正确值生成 inform 语句
                    connector = self.sample(["Oh no,", "Uhm sorry,", "Oh sorry,"])        # 连接语句
                    str_actions.append("%s %s %s" % (wrong_utt, connector, correct_utt))   # 错误语句连接正确语句
                else:
                    str_actions.append(get_inform_utt(slot_value))            # 否则只输出正确语句

            elif a.act == UserAct.CHAT:   # 闲聊，就随机采样一个 聊天的语句
                str_actions.append(self.sample(["What's your name?", "Where are you from?"]))

            elif a.act == UserAct.YN_QUESTION:                         # 如果用户是yes / no 的动作
                slot_type, expect_id = a.parameters[0]                 # 从参数中取出 slot name 和期望曹值的 id
                target_slot = self.domain.get_sys_slot(slot_type)      # 取出 系统slot对象
                expect_val = target_slot.vocabulary[expect_id]         # 取出 期望的val
                str_actions.append(target_slot.sample_yn_question(expect_val))   # 采样出yes no 问题模板

            elif a.act == UserAct.CONFIRM:
                str_actions.append(self.sample(["Yes.", "Yep.", "Yeah.", "That's correct.", "Uh-huh."]))

            elif a.act == UserAct.DISCONFIRM:
                str_actions.append(self.sample(["No.", "Nope.", "Wrong.", "That's wrong.", "Nay."]))

            elif a.act == UserAct.SATISFY:
                str_actions.append(self.sample(["No more questions.", "I have all I need.", "All good."]))

            elif a.act == UserAct.MORE_REQUEST:
                str_actions.append(self.sample(["I have more requests.", "One more thing.", "Not done yet."]))

            elif a.act == UserAct.NEW_SEARCH:
                str_actions.append(self.sample(["I want to search a new one.", "New request.", "A new search."]))

            else:
                raise ValueError("Unknown user act %s for NLG" % a.act)

        return " ".join(str_actions)     # 拼接各个action的语句

    def add_hesitation(self, sents, actions):
        pass

    def add_self_restart(self, sents, actions):
        pass
