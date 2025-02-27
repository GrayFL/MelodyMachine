import mido
import numpy as np
from typing import Union, Literal
from re import findall, IGNORECASE
from pathlib import Path

from ..base import AllNumType, AllStrType, AllIntType
from ..base import Chord
from ..config import SESSION_DATA


class MTracks(dict):
    '''
    ⚠️Warning: 不推荐使用该类，Song类具有更丰富的功能
    ---
    
    所有音符轨道的聚合，继承自dict

    类型: dict[str, NDarray]

    ---

    轨道格式:

    [[pitch, velocity, t_start, t_end, duration, midi_channel, channel_name],
     [  0  ,    1    ,    2   ,   3  ,    4    ,      5      ,      6], ...]
    '''

    def __init__(self, mid_fp: Union[str] = None, **kwds):
        super().__init__()
        self.bpM = SESSION_DATA.bpM
        self.Bar = SESSION_DATA.Bar
        self.InitBar = SESSION_DATA.InitBar
        if None is not mid_fp:
            self._parse_midifile(mid_fp)

        for k, v in kwds.items():
            if k in self.__dict__:
                self.__dict__[k] = v

    def _parse_midifile(self, mid: Union[str, mido.MidiFile]):
        if isinstance(mid, str):
            mid = mido.MidiFile(mid)
        else:
            pass
        for trk in mid.tracks:
            trk: mido.MidiTrack
            msg: mido.MetaMessage = trk[0]
            if 'set_tempo' == msg.type:
                self.bpM = round(60 / msg.tempo * 1000000, 1)
            if 'track_name' == msg.type:
                mtrack = self._process_miditrack(trk)
                if mtrack is not None:  # 非None
                    self[msg.name] = mtrack

    def _process_miditrack(self, track: mido.MidiTrack):
        t = 0  # tick time
        note_pool = {}
        mtrack = []
        for msg in track:
            t += msg.time
            if 'note_on' == msg.type:
                if msg.note in note_pool:
                    note_pool[msg.note].append([msg, t])  # [事件, 起始时间]
                else:
                    note_pool[msg.note] = [[msg, t]]
            elif 'note_off' == msg.type:
                if msg.note in note_pool:
                    [_msg, t_start] = note_pool[msg.note].pop(0)
                    mtrack.append([
                        _msg.note,  # pitch
                        _msg.velocity,
                        t_start,
                        t,  # t_end
                        t - t_start,  # duration
                        _msg.channel,
                        ])
                else:
                    raise RuntimeError('音符消息没有闭合')
        if len(mtrack) > 0:
            mtrack = np.array(mtrack, dtype=np.float32)
            mtrack[:, 2:5] = mtrack[:, 2:5] / self.Bar
            mtrack[:, 2:4] = mtrack[:, 2:4] + self.InitBar
            # 将开始和结束时间 + InitBar 以匹配手稿时间
            return mtrack
        else:
            return None

    def get_note_playing(
            self, tBar: float, channels: Union[str, list[str]] = 'omni'
        ):
        outs = []
        # 将单字符输入转变为可迭代的轨道名
        if isinstance(channels, str):
            if 'omni' == channels:
                channels = list(self.keys())
            else:
                channels = [channels]
        if isinstance(channels, list):
            for channel in channels:
                out = self[channel][(self[channel][:, 2] <= tBar)
                                    & (self[channel][:, 3] > tBar)]
                if len(out) > 0:
                    outs.append(out)
        else:
            raise RuntimeError('channels 不是正确的类型')
        # 节选当前时间戳所在的所有音符
        outs = np.concatenate(outs, axis=0) if len(outs) > 0 else None
        return outs


class Song():
    """
    channel_map : dict
        双向的字典，映射了从乐器编号和名称之间的关系
    instrument_map : dict
        单向的字典，仅仅是名字到编号的映射
    
    Notes
    ---
    输入索引的形式包括如下例子:
    1. `song[4.5]`
    2. `song[1.5:3.5]`
    3. `song['piano']`
    4. `song[[1, 2.5]]`
    4. `song[['piano', 'bass']]`
    5. `song[1.5:3.5, 'piano']`
    6. `song[['piano', 'bass'], [4,6,8]]`
    7. `song[[1,2,3, slice(6,7), slice(8,9), 14]], ['piano', 'bass']]`
    8. `song[[slice(1,1), slice(2,2),... slice(6,7), ...]], ['piano', 'bass']]`
    """

    def __init__(
            self,
            mid_fp: Union[str, Path] = None,
            spb: int = None,
            bpB: int = None,
            bpM: int = None,
            tpb: int = None,
            Bar: int = None,
            InitBar: int = None,
            **kwds
        ) -> None:
        '''
        Parameters
        ---
        mid_fp :
            midi 文件路径
        spb :
            steps per beat
        bpB :
            beats per Bar
        bpM :
            beats per Minute
        tpb :
            ticks per beat
        Bar :
            以 tick 为单位的长度
        InitBar :
            初始化
        '''
        self._track: np.ndarray = None
        self.spb = spb or SESSION_DATA.spb  # steps per beat
        self.bpB = bpB or SESSION_DATA.bpB  # beats per Bar
        self.bpM = bpM or SESSION_DATA.bpM  # beats per Minutes
        self.tpb = tpb or SESSION_DATA.tpb  # ticks per beat / timebase

        if None is InitBar:
            self.InitBar = SESSION_DATA.InitBar
        else:
            self.InitBar = InitBar

        ## 计算二级参数
        self.spB = self.spb * self.bpB
        self.BpM = self.bpM / self.bpB
        self.step = self.tpb // 4  # Fl studio中可视的最小单位长度 24
        self.beat = self.step * self.spb
        self.Bar = Bar or self.beat * self.bpB
        self.channel_map: dict[int, str] = {}
        self.instrument_map: dict[str, int] = {}
        self.display_backend: Literal['pandas', 'str'] = 'pandas'
        self.range: tuple[float, float]
        if None is not mid_fp:
            self._parse_midifile(mid_fp)
        self._update_properties()

    @property
    def track(self):
        """
        Notes
        ---
        通道定义：

        |     0     |      1       |       2        |      3       |      4       |        5         |           6            |
        |:---------:|:------------:|:--------------:|:------------:|:------------:|:----------------:|:----------------------:|
        | ``pitch`` | ``velocity`` | ``time_start`` | ``time_end`` | ``duration`` | ``midi_channel`` | ``instrument_channel`` |
        """
        return self._track

    @property
    def instruments(self):
        return self.get_instruments()

    @property
    def pitch_range(self):
        return self.get_pitch_range()

    @property
    def duration(self):
        return self.track[:, [4]]

    @property
    def range(self):
        return (self.track[:, 2].min(), self.track[:, 3].max())

    def _new(self, **kwds):
        new_ins = self.__new__(self.__class__)
        for k, v in self.__dict__.items():
            if k in kwds:
                new_ins.__dict__[k] = kwds[k]
            else:
                new_ins.__dict__[k] = v
        new_ins._update_properties()
        return new_ins

    def _update_properties(self):
        # self.range = (self.track[:, 2].min(), self.track[:, 3].max())
        ...

    def _parse_midifile(self, mid: Union[str, Path, mido.MidiFile]):
        if isinstance(mid, str):
            mid = mido.MidiFile(mid)
        elif isinstance(mid, Path):
            mid = mido.MidiFile(mid)
        else:
            pass
        _track = []
        _channel_id = 0
        for _i, trk in enumerate(mid.tracks):
            trk: mido.MidiTrack
            msg: mido.MetaMessage = trk[0]
            # if ('set_tempo' == msg.type) and (None is self.bpM):
            if 0 == _i:
                for msg in trk:
                    if 'set_tempo' == msg.type:
                        self.bpM = round(60 / msg.tempo * 1000000, 1)
                        break
                continue
            if 'track_name' == msg.type:
                self.channel_map[_channel_id] = msg.name
                # self.channel_map[msg.name] = _channel_id
                self.instrument_map[msg.name] = _channel_id
                sub_track = self._process_miditrack(trk, _channel_id)
                _channel_id += 1
                if sub_track is not None:  # 非None
                    _track.append(sub_track)
        _track = np.concatenate(_track, axis=0)
        _track = _track[_track[:, 2].argsort()]
        self._track = _track

    def _process_miditrack(self, track: mido.MidiTrack, channel_id: int):
        """
        将 MidiTrack 解析为 NDArray

        定义：  
        [[pit, velo, t_st, t_ed, dur, midi_chn, instr_chn]]
        """
        t = 0  # tick time
        note_pool = {}
        mtrack = []
        for msg in track:
            t += msg.time
            if msg.type not in ['note_on', 'note_off']:
                continue
            if (('note_on' == msg.type) and (msg.velocity > 0)
                    and (msg.note not in note_pool)):
                note_pool[msg.note] = [[msg, t]]
            elif (msg.note in note_pool):
                if (('note_on' == msg.type) and (msg.velocity > 0)):
                    note_pool[msg.note].append([msg, t])  # [事件, 起始时间]
                elif (('note_off' == msg.type) or
                        (('note_on' == msg.type) and (msg.velocity == 0))):
                    # note_pool[msg.note].append([msg, t])
                    [_msg, t_start] = note_pool[msg.note].pop(0)
                    _dur = t - t_start
                    _dur = _dur if _dur < 8 * self.Bar else 8 * self.Bar
                    mtrack.append([
                        _msg.note,  # pitch
                        _msg.velocity,
                        t_start,
                        t_start + _dur,  # t_end
                        _dur,  # duration
                        _msg.channel,
                        channel_id,
                        ])
                else:
                    raise RuntimeError('音符消息没有闭合')

        if len(mtrack) > 0:
            mtrack = np.array(mtrack, dtype=np.float32)
            mtrack[:, 2:5] = mtrack[:, 2:5] / self.Bar
            mtrack[:, 2:4] = mtrack[:, 2:4] + self.InitBar
            # 将开始和结束时间 + InitBar 以匹配手稿时间
            return mtrack
        else:
            return None

    def search_channels(self, keys: Union[str, list[str]]):
        """
        根据关键词查找匹配的所有乐器轨道

        Parameters
        ---
        keys : str | list[str]
            将输入的 str 以正则表达式的方式对所有的进行匹配
        """
        if isinstance(keys, str):
            keys = [keys]
        if isinstance(keys, list):
            mth = []
            # _S = ';'.join(list(self.instrument_map.keys()))
            _S = ';'.join(self.get_instruments())
            for k in keys:
                _f = findall(
                    fr'(?<=^|)([^;]*(?:{k})[^;]*?)(?=;|$)', _S, IGNORECASE
                    )
                print(_f)
                if None is not _f:
                    mth += _f
            return mth

    def get_note_playing(
            self,
            tBar: Union[float, slice, list[float]],
            channels: Union[
                str,
                int,
                Literal['omni'],
                list[Union[str, int]],
                ] = 'omni',
            **kwds
        ):
        """
        节选时间范围内的所有音符

        Parameters
        ---
        tBar :
            支持使用一个时间点或者slice范围
        """
        outs = []
        channels = self._cvt_chn_to_id(channels)
        channel_selection = np.in1d(self.track[:, 6], channels)
        # np.in1d 返回一个长度等于第一个参数数组的 bool 数组，
        # 用以表示其中每一位数是否在第二个参数数组中
        # 所以就是保证选择的乐器是在当前的 track 中的
        if (isinstance(tBar, list)
                and all(isinstance(item, slice) for item in tBar)):
            outs = []
            for _tBar in tBar:
                _tBar: slice
                st: float = _tBar.start if _tBar.start is not None else 0
                ed: float = \
                    _tBar.stop if _tBar.stop is not None else self.range[1]
                outs.append(
                    self.track[(self.track[:, 2] <= ed)
                                & (self.track[:, 3] > st)
                                & channel_selection]
                    )
            outs = np.concatenate(outs, axis=0)
            outs = outs if len(outs) > 0 else None
            return self._new(_track=outs)
        # 以下是旧的兼容性代码
        elif isinstance(tBar, AllNumType):
            outs = self.track[(self.track[:, 2] <= tBar)
                                & (self.track[:, 3] > tBar)
                                & channel_selection]
        elif isinstance(tBar, slice):
            st: float = tBar.start or 0
            ed: float = tBar.stop or max(self.track[:, 2])
            outs = self.track[(self.track[:, 2] <= ed)
                                & (self.track[:, 3] > st)
                                & channel_selection]
        elif (isinstance(tBar, list)
                and all(isinstance(item, AllNumType) for item in tBar)):
            outs = [
                self.track[(self.track[:, 2] <= _tBar)
                            & (self.track[:, 3] >= _tBar)
                            & channel_selection] for _tBar in tBar
                ]
            outs = np.concatenate(outs, axis=0)

        outs = outs if len(outs) > 0 else None
        return self._new(_track=outs)

    def quantify(
            self,
            level: Union[Literal['Bar', 'beat', 'step'], float] = 'beat',
            inplace=False
        ):
        """
        量化音符

        Parameters
        ---
        level :
            就近量化到哪个级别的小节线；其中，beat 是 4/4 拍的四分音符，6/8 拍的八分音符，step 一般是 beat 的 1/4 ；当 level 输入为小数时，则基于 step 表示量化到 level * step 的级别
        inplace :
            是否替换
        """
        if inplace:
            track = self.track
        else:
            track = self.track.copy()
        if 'Bar' == level:
            track[:, 2:5] = track[:, 2:5].round(0)
        elif 'beat' == level:
            track[:, 2:5] = (track[:, 2:5] * self.bpB).round(0) / self.bpB
        elif 'step' == level:
            track[:, 2:5] = (track[:, 2:5] * self.spB).round(0) / self.spB
        elif isinstance(level, AllNumType):
            track[:, 2:5] = (track[:, 2:5] * self.spB
                                / level).round(0) / self.spB * level
        if not inplace:
            return self._new(_track=track)

    def get_chord(
            self,
            quantization_level: Literal['Bar', 'beat', 'step'] = None
        ):
        """
        获取当前所有音符所构成的和弦

        Returns
        ---
        chord : Chord
        """
        if None is not quantization_level:
            tmp_song = self.quantify(level=quantization_level)
            return Chord(tmp_song.track[:, 0])
        else:
            return Chord(self.track[:, 0])

    def get_instruments(self):
        tmp = {
            self.channel_map[ch_id]: ch_id
            for ch_id in self.track[:, 6]
            }
        return list(tmp.keys())

    def get_pitch_range(self):
        return [self.track[:, 0].min(), self.track[:, 0].max()]

    def _to_dataframe(self):
        import pandas as pd
        df = pd.DataFrame(
            data=self.track,
            columns=[
                'pit', 'vel', 't_st', 't_ed', 't_dur', 'mid_ch', 'ch_id'
                ]
            )
        df['name'] = df['ch_id'].apply(lambda x: self.channel_map[x])
        df = df.astype({
            'pit': int,
            'vel': int,
            't_st': float,
            't_ed': float,
            't_dur': float,
            'mid_ch': int,
            'ch_id': int,
            'name': str
            })
        return df

    def _cvt_chn_to_id(
            self,
            channels: Union[
                str,
                int,
                Literal['omni'],
                list[Union[str, int]],
                ],
            **kwds
        ):
        """
        将混合轨道id和乐器名称的输入统一转换为轨道id
        """
        if 'omni' == channels:
            channels = list(self.instrument_map.values())
        elif isinstance(channels, AllStrType):
            channels = [channels]
        elif not hasattr(channels, '__getitem__'):
            channels = [channels]
        channels = [
            chn
            if isinstance(chn, AllIntType) else self.instrument_map[chn]
            # if isinstance(chn, AllStrType) else None
            for chn in channels
            ]
        # 如果是通道id，直接返回；如果是字符串，查找对应id
        return channels

    def __getitem__(self, idx):
        tBar, channels = self._handle_index(idx)
        return self.get_note_playing(tBar, channels)

    def _handle_index(self, idx):
        """
        能够处理多种索引方式的函数

        输入数据类型包括内置的 `str` `float` `int` `slice` `list` `tuple` 以及
        numpy 的 `str_` `number` `ndarray` 等

        输入数据的形式包括如下例子:
        1. `song[4.5]`
        2. `song[1.5:3.5]`
        3. `song['piano']`
        4. `song[[1, 2.5]]`
        4. `song[['piano', 'bass']]`
        5. `song[1.5:3.5, 'piano']`
        6. `song[['piano', 'bass'], [4,6,8]]`
        7. `song[[1,2,3, slice(6,7), slice(8,9), 14]], ['piano', 'bass']]`
        8. `song[[slice(1,1), slice(2,2),... slice(6,7), ...]], ['piano', 'bass']]`

        所有的数据最终都会被转换为上述例子中的最后一种形式，即 tuple 类型，且前为时间点，后为通道，所有数据格式统一为 python 内置的 float 和 str 。占位符中，时间使用 slice(None) 表示全部时间，通道使用 'omni' 表示全部通道
        """

        def _handle_single(idx):
            if isinstance(idx, AllNumType):  # 纯数字类型 #1
                tBar = [slice(float(idx), float(idx))]
                channels = None
            elif isinstance(idx, slice):  # 切片类型 #2
                tBar = [idx]
                channels = None
            elif isinstance(idx, AllStrType):  # 字符串类型 #3
                tBar = None
                channels = [idx]
            return tBar, channels

        def _handle_sequence(idxx):
            if isinstance(idxx, np.ndarray):
                assert idxx.ndim == 1, '索引必须为 1 维'
                idxx = idxx.tolist()
            if (isinstance(idxx, list)
                    and all(isinstance(item, (AllNumType, slice))
                            for item in idxx)):
                tBar = [
                    item if isinstance(item, slice) else
                    slice(float(item), float(item)) for item in idxx
                    ]
                channels = None
            elif (isinstance(idxx, list)
                    and any(isinstance(item, AllStrType)
                            for item in idxx)):
                tBar = None
                channels = idxx
            else:
                raise TypeError('索引类型错误')
            return tBar, channels

        if isinstance(idx, tuple):
            assert len(idx) == 2, '最多输入一个时间索引和一个通道索引'
            if isinstance(idx[0], (list, np.ndarray)):
                tBar_0, channels_0 = _handle_sequence(idx[0])
            else:
                tBar_0, channels_0 = _handle_single(idx[0])
            if isinstance(idx[1], (list, np.ndarray)):
                tBar_1, channels_1 = _handle_sequence(idx[1])
            else:
                tBar_1, channels_1 = _handle_single(idx[1])
            tBar = tBar_0 or tBar_1 or [slice(None)]
            channels = channels_0 or channels_1 or 'omni'
        elif isinstance(idx, (list, np.ndarray)):
            tBar, channels = _handle_sequence(idx)
            tBar = tBar or [slice(None)]
            channels = channels or 'omni'
        else:
            tBar, channels = _handle_single(idx)
            tBar = tBar or [slice(None)]
            channels = channels or 'omni'
        return tBar, channels

    def __repr__(self) -> str:

        if 'str' == self.display_backend:

            def get_s_msg(msg: np.ndarray):
                _s = f'{int(msg[0]):3} | {int(msg[1]):3} | {msg[2]:7.3f} | {msg[3]:7.3f} | {msg[4]:5.3f} | {int(msg[5]):6} | {int(msg[6]):5} | {self.channel_map[msg[6]]}'
                return _s

            _s_title = f'''{'pit':3} | {'vel':3} | {'t_st':7} | {'t_ed':7} | {'t_dur':5} | {'mid_ch':6} | {'ch_id'} | {'name'}\n'''
            if None is not self.track:
                _s = _s_title + '\n'.join([
                    f'{get_s_msg(msg)}' for msg in self.track
                    ])
            else:
                _s = _s_title
        elif 'pandas' == self.display_backend:
            df = self._to_dataframe()
            _s = df.__repr__()

        return _s

    def _repr_html_(self):
        if 'str' == self.display_backend:
            _s = None
        elif 'pandas' == self.display_backend:
            df = self._to_dataframe()
            _s = df._repr_html_()

        return _s
