import mido
import numpy as np
from typing import Union, Literal
from re import findall, IGNORECASE
from pathlib import Path

from ..base import AllNumType, AllStrType
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
                if mtrack is not None:         # 非None
                    self[msg.name] = mtrack

    def _process_miditrack(self, track: mido.MidiTrack):
        t = 0                                            # tick time
        note_pool = {}
        mtrack = []
        for msg in track:
            t += msg.time
            if 'note_on' == msg.type:
                if msg.note in note_pool:
                    note_pool[msg.note].append([msg, t]) # [事件, 起始时间]
                else:
                    note_pool[msg.note] = [[msg, t]]
            elif 'note_off' == msg.type:
                if msg.note in note_pool:
                    [_msg, t_start] = note_pool[msg.note].pop(0)
                    mtrack.append([
                        _msg.note,                       # pitch
                        _msg.velocity,
                        t_start,
                        t,                               # t_end
                        t - t_start,                     # duration
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

    # def __init__(self, mid_fp: Union[str, Path] = None, **kwds):
    #     self.track: np.ndarray = None
    #     self.bpM = SESSION_DATA.bpM
    #     # self.bpM: int = None
    #     self.Bar = SESSION_DATA.Bar
    #     self.InitBar = SESSION_DATA.InitBar
    #     self.channel_map = {}
    #     self.instrument_map = {}
    #     self.display_backend: Literal['pandas', 'str'] = 'pandas'
    #     self.range: list[float] = []

    #     for k, v in kwds.items():
    #         if k in self.__dict__:
    #             self.__dict__[k] = v

    #     if None is not mid_fp:
    #         self._parse_midifile(mid_fp)

    #     self._update_properties()

    # def _new(self, **kwds):
    #     new_ins = Song(None, **self.__dict__)

    #     for k, v in kwds.items():
    #         if k in new_ins.__dict__:
    #             new_ins.__dict__[k] = v

    #     new_ins._update_properties()

    #     return new_ins

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
        self.track: np.ndarray = None
        self.spb = spb or SESSION_DATA.spb     # steps per beat
        self.bpB = bpB or SESSION_DATA.bpB     # beats per Bar
        self.bpM = bpM or SESSION_DATA.bpM     # beats per Minutes
        self.tpb = tpb or SESSION_DATA.tpb     # ticks per beat / timebase

        if None is InitBar:
            self.InitBar = SESSION_DATA.InitBar
        else:
            self.InitBar = InitBar

        ## 计算二级参数
        self.spB = self.spb * self.bpB
        self.BpM = self.bpM / self.bpB
        self.step = self.tpb // 4      # Fl studio中可视的最小单位长度 24
        self.beat = self.step * self.spb
        self.Bar = Bar or self.beat * self.bpB
        self.channel_map = {}
        self.instrument_map = {}
        self.display_backend: Literal['pandas', 'str'] = 'pandas'
        self.range: list[float] = []
        if None is not mid_fp:
            self._parse_midifile(mid_fp)

        self._update_properties()

    def _new(self, **kwds):
        new_ins = self.__new__(self.__class__)
        for k, v in self.__dict__.items():
            if k in kwds:
                new_ins.__dict__[k] = kwds[k]
            else:
                new_ins.__dict__[k] = v
        new_ins._update_properties()

        # for k, v in self.__dict__.items():
        #     if k not in kwds:
        #         kwds[k] = v
        # new_ins = self.__init__(**kwds)
        return new_ins

    def _update_properties(self):
        self.range = [self.track[:, 2].min(), self.track[:, 3].max()]

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
                self.channel_map[msg.name] = _channel_id
                self.instrument_map[msg.name] = _channel_id
                sub_track = self._process_miditrack(trk, _channel_id)
                _channel_id += 1
                if sub_track is not None:      # 非None
                    _track.append(sub_track)
        _track = np.concatenate(_track, axis=0)
        _track = _track[_track[:, 2].argsort()]
        self.track = _track

    def _process_miditrack(self, track: mido.MidiTrack, channel_id: int):
        t = 0  # tick time
        note_pool = {}
        mtrack = []

        for msg in track:
            t += msg.time
            if 'note_on' == msg.type:
                if msg.note in note_pool:
                    note_pool[msg.note].append([msg, t]) # [事件, 起始时间]
                else:
                    note_pool[msg.note] = [[msg, t]]
            elif 'note_off' == msg.type:
                if msg.note in note_pool:
                    [_msg, t_start] = note_pool[msg.note].pop(0)
                    mtrack.append([
                        _msg.note,                       # pitch
                        _msg.velocity,
                        t_start,
                        t,                               # t_end
                        t - t_start,                     # duration
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
            tBar: Union[float, slice],
            channels: Union[str, int, list[str], list[int]] = 'omni'
        ):
        # if not isinstance():
        #     raise RuntimeError('channels 不是正确的类型')
        outs = []
        # 将单字符输入转变为可迭代的轨道名
        if isinstance(channels, Union[str, int]):
            if 'omni' == channels:
                channels = list(range(len(self.channel_map) // 2))
                # 因为channel_map是双向的，所以直接//2就可以得到乐器数量
            else:
                channels = [self.channel_map[channels]]
        if isinstance(channels, list):
            assert len(channels) != 0, "未选取有效轨道"
            if isinstance(channels[0], str):
                channels = [self.channel_map[c] for c in channels]
            channel_selection = np.in1d(self.track[:, 6], channels)
        if isinstance(tBar, AllNumType):
            outs = self.track[(self.track[:, 2] <= tBar)
                                & (self.track[:, 3] > tBar)
                                & channel_selection]
        elif isinstance(tBar, slice):
            st = tBar.start or 0
            ed = tBar.stop or max(self.track[:, 2])
            outs = self.track[(self.track[:, 2] <= ed)
                                & (self.track[:, 3] > st)
                                & channel_selection]
        elif isinstance(tBar, list):
            outs = [
                self.track[(self.track[:, 2] <= _tBar)
                            & (self.track[:, 3] >= _tBar)
                            & channel_selection] for _tBar in tBar
                ]
            outs = np.concatenate(outs, axis=0)

        # 节选当前时间戳所在的所有音符
        outs = outs if len(outs) > 0 else None
        # return outs
        return self._new(track=outs)

    def quantify(
            self,
            level: Literal['Bar', 'beat', 'step'] = 'beat',
            inplace=False
        ):
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

        if not inplace:
            return self._new(track=track)

    def get_chord(
            self,
            quantization_level: Literal['Bar', 'beat', 'step'] = None
        ):
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

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            tBar, channels = idx, 'omni'
        else:
            tBar, channels, *_ = *idx, 'omni',
        tBar: Union[AllNumType, list[AllNumType], slice]
        channels: Union[str, int, list[str], list[int]]
        return self.get_note_playing(tBar, channels)

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
