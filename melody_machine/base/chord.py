import numpy as np
import pandas as pd
from typing import Union, Literal
from itertools import product, chain
from pathlib import Path
from re import findall
from .common import DEGREE_MAP, INTERVAL_MAP
from .base_types import AllStrType, AllNumType, NotationType, PitchType

from . import note

print(Path('chord_db.csv').absolute())
CHORD_DB = pd.read_csv('chord_db.csv')


class Chord():

    def __init__(
            self,
            notes: Union[
                list[Union[AllStrType]],  # 音符
                list[Union[AllNumType]],  # 音高
                ],
            ntype: Literal['default', 'pitch', 'notation'] = 'default'
        ):
        self.notes = None
        _notes = [None for _ in range(len(notes))]
        if 'default' == ntype:  # 根据第一个元素推断
            for _note in notes:
                break
            if isinstance(_note, AllStrType):
                ntype = 'notation'
            elif isinstance(_note, AllNumType):
                ntype = 'pitch'
        if 'notation' == ntype:  # 音符
            ntype = NotationType
            for i, _note in enumerate(notes):
                if isinstance(_note, AllNumType):
                    _note = note.pit2nta(_note)
                _notes[i] = _note
        elif 'pitch' == ntype:  # 音高
            ntype = PitchType
            for i, _note in enumerate(notes):
                if isinstance(_note, AllStrType):
                    _note = note.nta2pit(_note)
                _notes[i] = _note
        self.notes = np.array(_notes, dtype=ntype)

    def astype(self, ntype: Literal['pitch', 'notation']):
        return Chord(self.notes, ntype=ntype)

    def detect_chord(self, result_shift=0):
        '''
        chord检测器

        Parameters
        ---
        batch_chord: np.ndarray
            - 维度 [Batch size, amount of Notes]
        '''
        arr_intervals = self.notes - self.notes[..., None]
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
            chord_data = {'default': self.notes}
            chord_name = f'Unknown'
        else:
            index = result_shift
            id_best_match = arr_matched_id[index][0]  # 输入和弦中匹配的id
            root_note = note.pit2nta(
                self.notes[id_best_match], is_root_only=True
                )
            base_note = note.pit2nta(self.notes[0], is_root_only=True)
            inversion_mark = '' if base_note == root_note else f' /{base_note}'
            chord_data = CHORD_DB.loc[arr_matched_id[index][1]]
            chord_name = f'{root_note}{chord_data["和弦标记"]}{inversion_mark}'
        detection = {
            'chord name': chord_name,
            'root note': root_note,
            'chord data': chord_data,
            }
        return detection

    def __repr__(self) -> str:
        _s = f'{self.notes}'
        return _s


class Progression():

    def __init__(self, arr_chord: list[Chord]):
        self.chords = arr_chord

    def astype(self, ntype: Literal['pitch', 'notation']):
        return Progression([
            chord.astype(ntype=ntype) for chord in self.chords
            ])

    def detect_chord(self, result_shift=0):
        return [chord.detect_chord(result_shift) for chord in self.chords]

    def __repr__(self) -> str:
        _s = f'{self.chords}'
        return _s


def progression(
        arr_chord: list[list[Union[AllStrType, AllNumType]]],
        ntype: Literal['default', 'pitch', 'notation'] = 'default'
    ):
    return Progression([Chord(chord, ntype=ntype) for chord in arr_chord])


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


def degree2div(lst: Union[list[np.int8], list[str]]):
    '''
    从既有音程推测可能的等效音程组合
    '''
    if isinstance(lst[0], np.int8):
        lst = [str(i) for i in lst]
    else:
        pass
    return set(lst)


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


if '__main__' == __name__:
    a = [1, 2, 4]
    a = Chord(a)
    print(a)
    a = a.astype(ntype='notation')
    print(a)
