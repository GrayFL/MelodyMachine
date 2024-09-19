import mido
import numpy as np
from typing import Union
from sklearn.cluster import DBSCAN


class MTracks(dict):

    def __init__(self):
        super().__init__()
        self.bpM = 120

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
                mtrack = self.process_miditrack(trk)
                if mtrack is not None:  # 非None
                    self[msg.name] = mtrack

    def process_miditrack(self, track: mido.MidiTrack):
        '''
        !Warning 手稿的时间戳建议按照FL的小节记数，即1,2,...  
        MidiPattern和Paragraph将统一采用这个计数法，因此在处理Midi事件的时间标记时，会统一 + InitBar 来匹配小节时间

        而对于显示时间，也默认从InitBar开始。详见InitBar定义
        '''
        t = 0  # tick time
        note_pool = {}
        mtrack = []
        for msg in track:
            t += msg.time
            if 'note_on' == msg.type:
                note_pool[msg.note] = [msg.velocity, t]
            elif 'note_off' == msg.type:
                if msg.note in note_pool:
                    [value, t_start] = note_pool.pop(msg.note)
                    mtrack.append([
                        msg.note,  # 音高
                        value,  # 力度
                        t_start,  # 开始时间
                        t,  # 结束时间
                        t - t_start,  # 持续长度
                        ])
        if len(mtrack) > 0:
            mtrack = np.array(mtrack, dtype=np.float32)
            # mtrack[:, 2:5] = mtrack[:, 2:5] / self.Bar
            # mtrack[:, 2:4] = mtrack[:, 2:4] + self.InitBar
            # 将开始和结束时间 + InitBar 以匹配手稿时间
            return mtrack
        else:
            return None
