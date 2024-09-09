import numpy as np
from typing import Union
from re import search
from .common import NOTE_MAP, SHIFT_MAP, PITCH_MAP, DEFAULT_OCT


def notation2pitch(notation: Union[str, np.str_], default_oct=DEFAULT_OCT):
    '''
    将**音符标记**格式转换为**音高**格式
    '''
    res = search('([A-G])\s*([#b]*)\s*(\d*)', notation)
    # res.group(0) 是 整个匹配的字符串
    root = res.group(1)
    shift = res.group(2)
    oct = res.group(3) or default_oct
    pitch = NOTE_MAP[root] + SHIFT_MAP[shift] + int(oct) * 12
    return pitch


def pitch2notation(
        pitch: Union[int, float, np.number], is_root_only=False
    ):
    '''
    将**音高**格式转换为**音符标记**格式
    '''
    if is_root_only:
        root = pitch % 12
        note = PITCH_MAP[root]
    else:
        root = pitch % 12
        oct = pitch // 12
        note = PITCH_MAP[root] + str(oct)
    return note


nta2pit = notation2pitch
pit2nta = pitch2notation
