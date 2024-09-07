import numpy as np
import pandas as pd
from typing import Union
from re import findall

NOTE_MAP = {
    # dict[str,int]
    'C': 0,
    'D': 2,
    'E': 4,
    'F': 5,
    'G': 7,
    'A': 9,
    'B': 11,
    }

SHIFT_MAP = {
    # 关于增减有一些麻烦，尚且有些bug
    '': 0,  # 还原记号或无升降记号
    'b': -1,
    '#': 1,
    'P': 0,  # 纯
    'M': 0,  # 大
    'm': -1,  # 小
    'd': -1,  # 减
    'A': +1,  # 增
    }

DEGREE_MAP = {
    # 音的度数
    # 如9度 9/7=1...2，即等效2度音
    # numerical name 度数
    # 以自然大调为基准
    0: 0,  # 纯一度
    1: 2,  # 大二度
    2: 4,  # 大三度
    3: 5,  # 纯四度
    4: 7,  # 纯五度
    5: 9,  # 大六度
    6: 11,  # 大七度
    }

INTERVAL_MAP = {
    # 音程数
    0: '1P',  # 纯一度/纯八度
    1: '2m',  # 小二度
    2: '2M',  # 大二度
    3: '3m',  # 小三度
    4: '3M',  # 大三度
    5: '4P',  # 纯四度
    6: '4A',  # 增四度/减五度(5d)
    7: '5P',  # 纯五度
    8: '6m',  # 小六度/增五度(5A)
    9: '6M',  # 大六度
    10: '7m',  # 小七度
    11: '7M',  # 大七度
    }

PITCH_MAP = {
    # dict[int,str]
    0: 'C',
    1: 'C#',
    2: 'D',
    3: 'D#',
    4: 'E',
    5: 'F',
    6: 'F#',
    7: 'G',
    8: 'G#',
    9: 'A',
    10: 'A#',
    11: 'B',
    }

# CHORD_MAP = {''}
CHORD_DB = pd.read_csv('chord_db.csv')
CHORD_DB.set_index(CHORD_DB['音程'], inplace=True)
# CHORD_DB.to_csv('chord_db.csv', index=False)


@staticmethod
def note2pitch(note: Union[str, list[str], np.ndarray], default_oct=5):
    '''
    从字符串形式的note转换到midi协议的音高形式

    Parameters
    ---
    note: (str|list|ndarray)
        - 如 `'C'` `'G#4'` `Fb6` `'D10'`
    default_oct: (int) = 5
        - 在没有指定八度时的默认八度
    
    Return
    ---
    pitch: (int)
    '''
    if isinstance(note, str):
        root, shift, oct = findall(
            r'^\s*(\S)\s*([#b])*\s*(\d*)\s*$', note
            )[0]
        if '' == oct:
            oct = default_oct
        pitch = NOTE_MAP[root] + SHIFT_MAP[shift] + int(oct) * 12
        return pitch
    elif isinstance(note, Union[list, np.ndarray]):
        kwds = {'default_oct': default_oct}
        return [
            chord_p([note2pitch(n, **kwds) for n in chord])
            for chord in note
            ]


@staticmethod
def pitch2note(
        pitch: Union[np.int8, list[np.int8], np.ndarray],
        is_root_only=False
    ):
    '''
    从midi协议的音高形式的note转换到字符串形式

    Parameters
    ---
    pitch: str|list|ndarray
        - 如 `11` `[11,23]`
    is_root_only: bool = False
        - 输出时是否包含八度标记
    
    Return
    ---
    note: str
    '''
    if isinstance(pitch, np.int8):
        root, oct = pitch % 12, pitch // 12
        if is_root_only:
            note = PITCH_MAP[root]
        else:
            note = PITCH_MAP[root] + str(oct)
        return note
    elif isinstance(pitch, Union[list, np.ndarray]):
        kwds = {'is_root_only': is_root_only}
        return [
            chord_s([pitch2note(n, **kwds) for n in chord])
            for chord in pitch
            ]


@staticmethod
def degree2interval(
        degrees: Union[str, list[str], list[tuple[str, str]]],
        is_sort=False,
        is_remove_duplicates=False
    ):
    '''
    从“度”转换为“音程”。如大三度=>4个半音

    Parameters
    ---
    degrees:
        - 和弦的度数序列，如
        `'1P,5P,7m,9M,11P'`
        `['1P','5P','7m','9M','11P']`
        `[('1', 'P'), ('5', 'P'), ('7', 'm'), ('9', 'M'), ('11', 'P')]`
    '''
    if isinstance(degrees, str):
        degrees = findall('(\d+)(\S)', degrees)
    elif isinstance(degrees, list):
        if isinstance(degrees[0], str):
            degrees = findall(
                '(\d+)(\S)', ''.join(['1P', '5P', '7m', '9M', '11P'])
                )
        elif isinstance(degrees[0], tuple):
            pass
    intervals = [
        DEGREE_MAP[(np.int8(degree) - 1) % 7] + SHIFT_MAP[shift]
        for (degree, shift) in degrees
        ]
    if is_remove_duplicates:
        intervals = list(dict.fromkeys(intervals))
        # intervals = list(set(intervals))
        # set的方式会重排，fromkeys在py3.7后保留插入顺序
    if is_sort:
        intervals = sorted(intervals)
    return intervals


@staticmethod
def interval2degree(
        intervals: Union[str, list[int], np.ndarray],
        is_sort=False,
        is_remove_duplicates=False
    ):
    if isinstance(intervals, str):
        intervals = findall('(\d+)', intervals)
        intervals = [np.int8(interval) for interval in intervals]
    else:
        pass
    degrees = [INTERVAL_MAP[interval] for interval in intervals]
    if is_remove_duplicates:
        degrees = list(dict.fromkeys(degrees))
        # degrees = list(set(degrees))
        # set的方式会重排，fromkeys在py3.7后保留插入顺序
    if is_sort:
        degrees = sorted(degrees)
    return degrees


@staticmethod
def list2div(
        lst: Union[list[np.int8], list[str]],
        is_sort=False,
        is_remove_duplicates=False
    ):
    if isinstance(lst[0], np.int8):
        lst = [str(i) for i in lst]
    else:
        pass
    if is_remove_duplicates:
        lst = list(dict.fromkeys(lst))
    if is_sort:
        lst = sorted(lst)
    return ','.join(lst)


@staticmethod
def detect_chord(batch_chord: list[np.ndarray]):
    '''
    chord检测器

    Parameters
    ---
    batch_chord: np.ndarray
        - 维度 [Batch size, amount of Notes]
    '''
    detections = []
    for chord in batch_chord:
        arr_intervals = chord - chord[..., None]
        # 相当于求以每个组成音作为根音时其他音的相对音程
        arr_intervals = arr_intervals % 12
        arr_intervals.sort(axis=1)
        div_intervals = [
            list2div(intervals, is_remove_duplicates=True)
            for intervals in arr_intervals
            ]
        div_best_match = None  # 匹配和弦的音程关系
        id_best_match = 0  # 匹配和弦的序号
        priority = -1  # 和弦库中的匹配优先级
        for i, div in enumerate(div_intervals):
            try:
                if (div in CHORD_DB.index
                    ) and (CHORD_DB.loc[div, '优先级'] > priority):
                    div_best_match = div
                    id_best_match = i
                    priority = CHORD_DB.loc[div, '优先级']
            except Exception as e:
                print(e)
                print(div)
                print(CHORD_DB.loc[div, '优先级'])
        if None is div_best_match:
            root_note = None
            chord_data = {'default': chord}
            chord_name = f'Unknown'
        else:
            root_note = pitch2note(chord[id_best_match], is_root_only=True)
            chord_data = CHORD_DB.loc[div_best_match]
            chord_name = f'{root_note}{chord_data["和弦标记"]}'
        detections.append({
            'chord name': chord_name,
            'root note': root_note,
            'chord data': chord_data,
            })

    return detections


@staticmethod
def chord_s(chord):
    '''
    生成字符串('U4')型chord序列

    Parameter
    ---
    chord: (同ndarray)
        - 维度为1的和弦序列
        形如 `['C4', 'E4', 'G4']`

    Return
    ---
    dtype为'U4'的ndarray
    '''
    return np.array(chord, dtype='U4')


@staticmethod
def chord_p(chord):
    '''
    生成音高(np.int8)型chord序列

    Parameter
    ---
    chord: (同ndarray)
        - 维度为1的和弦序列
        形如 `[48, 52, 55]`

    Return
    ---
    dtype为np.uint8的ndarray
    '''
    return np.array(chord, dtype=np.int8)


@staticmethod
def chord_progression(
    arr_chord: Union[list[list[np.int8]], list[list[int]],
                        list[list[np.str_]], list[list[str]]]
    ):
    if isinstance(arr_chord[0][0],
                    int) or isinstance(arr_chord[0][0], np.int8):
        progression = [chord_p(chord) for chord in arr_chord]
    elif isinstance(arr_chord[0][0],
                    str) or isinstance(arr_chord[0][0], np.str_):
        progression = [chord_s(chord) for chord in arr_chord]
    return progression


class Visualizer:
    ...
