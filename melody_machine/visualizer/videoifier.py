from .base_visualizer import BaseVisualizer

import numpy as np
from ..miditools import Song, MTracks
from .base_types import COLOR, KEYBOARD
import moviepy as me
import matplotlib.pyplot as plt


class MovieVisualizer(BaseVisualizer):
    '''
    MovieVisualizer基于MoviePy增加了可动小节线
    '''

    def __init__(self) -> None:
        super().__init__()

    def play(self, song: Song):
        plt.close()
        fig, ax_bg, ax_fg, art_timeline = BaseVisualizer._init_figure(
            self.h / 1080
            )
        self.make_fig(song, fig, ax_bg, ax_fg)
        xlim = song.range
        art_timeline, *_ = ax_fg.plot(
            [],
            [],
            color=COLOR.COLOR_TIME_LINE,
            zorder=3,
            )

        def makeFrame(t: float):
            t_bar = (t/60) * self.BpM + xlim[0]
            art_timeline.set_data([t_bar, t_bar], [0, 102])
            fig.canvas.draw()
            frame = np.array(fig.canvas.buffer_rgba())
            return frame[:, :, :3]

        vc = me.VideoClip(
            makeFrame,
            duration=self.spanBar2True(xlim[1] - xlim[0]),
            )
        return vc

    def timeBar2Trk(self, tBar: float):
        '''
        从Bar时间换算到音乐时间
        '''
        return (tBar - self.InitBar) / self.BpM * 60

    def timeTrk2Bar(self, tTrk: float):
        '''
        从音乐时间换算到Bar时间
        '''
        return tTrk / 60 * self.BpM + 1

    def timeBar2Mov(self, tBar: float):
        '''
        从Bar时间换算到视频时间
        '''
        return (tBar - self.InitBar) / self.BpM * 60 + self.BeginTime

    def timeMov2Bar(self, tMov: float):
        '''
        从视频时间换算到Bar时间
        '''
        return (tMov - self.BeginTime) / 60 * self.BpM + self.InitBar

    def spanBar2True(self, span_tBar: float):
        '''
        将Bar时间跨度换算到真实事件跨度
        '''
        return span_tBar / self.BpM * 60
