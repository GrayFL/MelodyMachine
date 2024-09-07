import numpy as np
import pandas as pd
from typing import Union
from itertools import product, chain
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
    '1P': 0,  # 纯一度
    '2m': 1,  # 小二度
    '2M': 2,  # 大二度
    '3m': 3,  # 小三度
    '3M': 4,  # 大三度
    '4P': 5,  # 纯四度
    '4A': 6,  # 增四度
    '5d': 6,  # 减五度
    '5P': 7,  # 纯五度
    '6m': 8,  # 小六度
    '5A': 8,  # 增五度
    '6M': 9,  # 大六度
    '7d': 9,  # 大六度
    '7m': 10,  # 小七度
    '7M': 11,  # 大七度
    '8P': 12,  # 纯八度
    '9m': 13,  # 小九度（小二度）
    '8A': 13,  # 增八度
    '9M': 14,  # 大九度（大二度）
    '9A': 15,  # 增九度（增二度）
    '10m': 15,  # 小十度（小三度）
    '10M': 16,  # 大十度（大三度）
    '11P': 17,  # 纯十一度（纯四度）
    '11A': 18,  # 增十一度（增四度）
    '12d': 18,  # 减十二度（减五度）
    '12P': 19,  # 纯十二度（纯五度）
    '13m': 20,  # 小十三度（小六度）
    '13M': 21,  # 大十三度（大六度）
    '14m': 22,  # 小十四度（小七度）
    '14M': 23,  # 大十四度（大七度）
    }

INTERVAL_MAP = {
    # 音程数
    0: ['1P'],  # 纯一度
    1: ['2m', '9m'],  # 小二度（小九度）
    2: ['2M', '9M'],  # 大二度（大九度）
    3: ['3m', '10m'],  # 小三度（小十度）
    4: ['3M', '10M'],  # 大三度（大十度）
    5: ['4P'],  # 纯四度
    6: ['4A', '5d'],  # 增四度/减五度
    7: ['5P'],  # 纯五度
    8: ['6m', '5A', '13m'],  # 小六度（小十三度）/增五度
    9: ['6M', '7d', '13M'],  # 大六度（大十三度）
    10: ['7m', '14m'],  # 小七度（小十四度）
    11: ['7M', '14M'],  # 大七度（大十四度）
    12: ['8P', '1P'],  # 纯八度（纯一度）
    13: ['9m', '8A', '2m'],  # 小九度（小二度）/增八度
    14: ['9M', '2M'],  # 大九度（大二度）
    15: ['10m', '9A', '3m'],  # 小十度（小三度）/增九度（增二度）
    16: ['10M', '3M'],  # 大十度（大三度）
    17: ['11P', '4P'],  # 纯十一度（纯四度）
    18: ['11A', '12d', '4A', '5d'],  # 增十一度（增四度）/减十二度（减五度）
    19: ['12P', '5P'],  # 纯十二度（纯五度）
    20: ['13m', '6m'],  # 小十三度（小六度）
    21: ['13M', '6M'],  # 大十三度（大六度）
    22: ['14m', '7m'],  # 小十四度（小七度）
    23: ['14M', '7M'],  # 大十四度（大七度）
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
# CHORD_DB.set_index(CHORD_DB['度数'], inplace=True)
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
        degrees: Union[str, list[str]],
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
    '''
    if isinstance(degrees, str):
        degrees = findall('(\d+\S)', degrees)
    elif isinstance(degrees, list):
        pass
    intervals = [DEGREE_MAP[degree] for degree in degrees]
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
    '''
    从“音程”转换为“度”。如4个半音=>大三度

    Parameters
    ---
    intervals:
        - 和弦的度数序列，如
        `'0,4,6,9'`
        `[0,4,6,9]`
    
    Return
    ---
    arr_degrees:
        - `[('1P', '3M', '4A', '6M'),
            ('1P', '3M', '4A', '7d'), ...]`
    '''
    if isinstance(intervals, str):
        intervals = findall('(\d+)', intervals)
        intervals = [np.int8(interval) for interval in intervals]
    else:
        pass
    if is_remove_duplicates:
        intervals = list(dict.fromkeys(intervals))
        # intervals = list(set(intervals))
        # set的方式会重排，fromkeys在py3.7后保留插入顺序
    if is_sort:
        intervals = sorted(intervals)
    args = (INTERVAL_MAP[interval] for interval in intervals)
    arr_degrees = list(product(*args))
    return arr_degrees


@staticmethod
def degree2div(lst: Union[list[np.int8], list[str]]):
    '''
    从既有音程推测可能的等效音程组合
    '''
    if isinstance(lst[0], np.int8):
        lst = [str(i) for i in lst]
    else:
        pass
    return set(lst)


@staticmethod
def find_matched_id(arr_shift_degrees: list[list[tuple[str]]]):
    '''
    给出组成音，查找包含这些音的所有和弦，并按音数和优先级排序。
    其中音数越少、优先级越大越靠前

    Parameters
    ---
    arr_shift_degrees: list[list[tuple[str]]]
        - 和弦集合，形如 `[[('1P', '3M', '5P')], [('13m', '1P', '3m')], ...]`
    
    Return
    ---
    排序好的和弦查询id列表
    '''
    arr_matched_id = []
    for input_id, shift_degrees in enumerate(arr_shift_degrees):
        # input_id 指输入序列中的序号，即以第几个为根音
        for degrees in shift_degrees:
            div = degree2div(degrees)
            for database_id, s in enumerate(CHORD_DB['度数']):
                # database_id 指数据库中的序号
                if div <= set(s.split(',')):
                    arr_matched_id.append((input_id, database_id))
    arr_matched_id = list(dict.fromkeys(arr_matched_id))

    arr_matched_id = sorted(
        arr_matched_id,
        key=lambda i: 100 * CHORD_DB.loc[i[1], '组成音数'] \
                          - CHORD_DB.loc[i[1], '优先级'],
        )  # 音数越少、优先级越大越靠前
    # print(
    #     CHORD_DB.loc[[matched_id[1] for matched_id in arr_matched_id],
    #                     ['和弦标记', '优先级']]
    #     )
    return arr_matched_id


@staticmethod
def detect_chord(batch_chord: list[np.ndarray], result_shift=0):
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
        arr_intervals = arr_intervals % 24
        # 量化到两个八度内
        arr_shift_degrees = list([
            interval2degree(intervals) for intervals in arr_intervals
            ])
        # print(arr_shift_degrees)
        arr_matched_id = find_matched_id(arr_shift_degrees)
        if len(arr_matched_id) == 0:  # 和弦库未收录
            root_note = None
            base_note = None
            chord_data = {'default': chord}
            chord_name = f'Unknown'
        else:
            index = result_shift
            id_best_match = arr_matched_id[index][0]  # 输入和弦中匹配的id
            root_note = pitch2note(chord[id_best_match], is_root_only=True)
            base_note = pitch2note(chord[0], is_root_only=True)
            inversion_mark = '' if base_note == root_note else f' /{base_note}'
            chord_data = CHORD_DB.loc[arr_matched_id[index][1]]
            chord_name = f'{root_note}{chord_data["和弦标记"]}{inversion_mark}'
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
