import numpy as np
import pandas as pd
from typing import Union, Literal
from itertools import product, chain
from pathlib import Path
from re import findall
from .common import DEGREE_MAP, INTERVAL_MAP
from .base_types import AllStrType, AllNumType, AllIntType, NotationType, PitchType

from . import note


def find_library_path(library_name):
    import importlib.util
    spec = importlib.util.find_spec(library_name)
    library_path = Path(spec.origin).parent
    return library_path


library_path = find_library_path('melody_machine')
print((library_path / 'chord_db.csv').absolute())
CHORD_DB = pd.read_csv(library_path / 'chord_db.csv')


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
        self.ntype = None

        _notes = [None for _ in range(len(notes))]
        if 'default' == ntype:  # 根据第一个元素推断
            for _note in notes:
                break
            if isinstance(_note, AllStrType):
                ntype = 'notation'
            elif isinstance(_note, AllNumType):
                ntype = 'pitch'
        if 'notation' == ntype:  # 音符
            self.ntype = 'notation'
            ntype = NotationType
            for i, _note in enumerate(notes):
                if isinstance(_note, AllNumType):
                    _note = note.pit2nta(_note)
                _notes[i] = _note
        elif 'pitch' == ntype:  # 音高
            self.ntype = 'pitch'
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
        batch_chord : np.ndarray
            - 维度 [Batch size, amount of Notes]
        '''
        # 转换为pitch类型
        if 'notation' == self.ntype:
            notes = notation2pitch(self.notes)
        else:
            notes = self.notes
        arr_intervals = notes - notes[..., None]
        # 相当于求以每个组成音作为根音时其他音的相对音程
        arr_intervals = arr_intervals % 24
        # 量化到两个八度内
        arr_shift_degrees = list([
            interval2degree(intervals, is_remove_duplicates=True)
            for intervals in arr_intervals
            ])
        arr_matched_id = find_matched_id(arr_shift_degrees)
        if len(arr_matched_id) == 0:  # 和弦库未收录
            root_note = None
            base_note = None
            chord_data = {'default': notes}
            chord_name = f'Unknown'
        else:
            index = result_shift
            id_best_match = arr_matched_id[index][0]
            # id_best_match 是匹配最好的根音候选者序号
            root_note = note.pit2nta(
                notes[id_best_match], is_root_only=True
                )
            base_note = note.pit2nta(notes[0], is_root_only=True)
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
        _s = f'Chord({self.notes})'
        return _s


def pitch2notation(notes: list[AllNumType]):
    _notes = [None for _ in range(len(notes))]
    for i, _note in enumerate(notes):
        if isinstance(_note, AllNumType):
            _note = note.pit2nta(_note)
        _notes[i] = _note
    return np.array(_notes, dtype=NotationType)


def notation2pitch(notes: list[AllStrType]):
    _notes = [None for _ in range(len(notes))]
    for i, _note in enumerate(notes):
        if isinstance(_note, AllStrType):
            _note = note.nta2pit(_note)
        _notes[i] = _note
    return np.array(_notes, dtype=PitchType)


nta2pit = notation2pitch
pit2nta = pitch2notation


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
    degrees :
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


def interval2degree(
        intervals: Union[str, list[int], np.ndarray],
        is_sort=False,
        is_remove_duplicates=False
    ):
    '''
    从“音程”转换为“度”。如4个半音=>大三度

    Parameters
    ---
    intervals :
        - 和弦的度数序列，如  
        `'0,4,6,9'`  
        `[0,4,6,9]`
    
    Return
    ---
    arr_degrees :
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
    # print('\n'.join([f'{shift_degrees}' for shift_degrees in arr_degrees]))
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


def find_matched_id(shift_alt_degrees: list[list[tuple[str]]]):
    '''
    给出组成音，查找包含这些音的所有和弦，并按音数和优先级排序。
    其中音数越少、优先级越大越靠前

    Parameters
    ---
    shift_alt_degrees: list[list[tuple[str]]]
        - 和弦集合，形如 `[[('1P', '3M', '5P')], [('13m', '1P', '3m')], ...]`
        shift指以不同音作为根音，形似转位  
        alt指不同可能的音程组合
        degrees指音程组合
    
    Return
    ---
    排序好的和弦查询id列表
    '''
    # arr_matched_id = []
    dic_score = {}
    len_inv = len(shift_alt_degrees)
    for input_id, alt_degrees in enumerate(shift_alt_degrees):
        # input_id 指输入序列中的序号，即以第几个为根音
        for degrees in alt_degrees:
            div = degree2div(degrees)
            # print('innnnnnnnnnn', div)
            for database_id in range(len(CHORD_DB)):
                # database_id 指数据库中的序号
                ref = CHORD_DB.loc[database_id]
                # 计算相似度
                score = get_score(div, ref, input_id, len_inv)
                # print(div, set(s.split(',')), score_similarity)
                if score >= 0.1:
                    if ((input_id, database_id) not in dic_score) or (
                        (input_id, database_id) in dic_score
                            and score > dic_score[(input_id, database_id)]
                        ):
                        dic_score[(input_id, database_id)] = score

    arr_matched_id = sorted(
        dic_score.keys(), key=lambda x: dic_score[x], reverse=True
        )
    # print([(k[0], CHORD_DB.loc[k[1], '和弦标记'], dic_score[k])
    #         for k in arr_matched_id])
    return arr_matched_id


def get_score(
        div: set, ref: Union[pd.Series, dict], inv: int, len_inv: int
    ):
    '''
    Parameters
    ---
    div:
        - 提取特征音程集合
    ref:
        - 查表参考
    inv:
        - 以第几个音作为根音
    len_inv:
        - 总转位数量
    '''
    len_ = (ref['组成音数'] + len(div)) / 2
    priority = ref['优先级']
    deg = set(ref['度数'].split(','))

    similarity = len(div & deg) / len_

    # print(div, deg, priority, similarity)
    score = similarity*0.6 + ((priority+10) / 110) * 0.4
    # score = similarity
    # print(score)
    return score


if '__main__' == __name__:
    a = [1, 2, 4]
    a = Chord(a)
    print(a)
    a = a.astype(ntype='notation')
    print(a)
