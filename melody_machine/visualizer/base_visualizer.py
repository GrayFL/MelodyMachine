import numpy as np
from ..miditools import Song, MTracks
from .base_types import COLOR, KEYBOARD
from ..config import SESSION_DATA
import matplotlib.pyplot as plt


class MidiPattern():
    '''
    Midi段落类
    '''

    def __init__(
            self,
            midi_range: list[float],
            channels: dict = {},
            para_range: list[float] = None,
            pitch_clip_range: list[float] = None
        ) -> None:
        '''
        Parameters
        ---
        midi_range:
            - Midi段落的持续范围，其计量单位是小节时间 `tBar`  
            形如 [1, 5] ，表示小节时间1到5的范围，长度为4
        channels:
            - 选择的乐器轨道  
            形如 {'str vln #1': 'Violin I'}  
            前者为Midi文件中的命名，后者为显示时的替换名称
        para_range:
            - 该Midi段落所关联的文本段落的**总**范围
        pitch_clip_range:
            - 该Midi段落的音高最大范围（选择音符时会过滤掉超出该范围的音高，但不代表最终的显示范围）
        
        Properties
        ---
        disp_range:
            - 该Midi段落的总显示范围，即汇总后的para_range
        mtracks:
            - 该Midi段落所包含的所有音符，且按轨道名称分类
        '''
        self.range = midi_range
        self.channels = channels
        self.disp_range = para_range
        self.pitch_clip_range = pitch_clip_range
        self.mtracks: dict[str, np.ndarray] = None
        self.pitch_range: list[float] = None

    def get_div_id(self) -> str:
        '''
        生成用于唯一化标识Midi段落的字符串ID
        '''
        return f'{self.range}{self.channels}'

    def update_disp_range(self, para_range: list[float]):
        '''
        更新Midi段落的显示范围（取外边界）
        '''
        if para_range[1] > self.disp_range[1]:
            self.disp_range[1] = para_range[1]
        if para_range[0] < self.disp_range[0]:
            self.disp_range[0] = para_range[0]

    def __repr__(self) -> str:
        _s = f'''
== MidiPattern =====================================
range           : {self.range}
disp_range      : {self.disp_range}
channels        : {self.channels}
pitch_clip_range: {self.pitch_clip_range}
pitch_range     : {self.pitch_range}
mtracks         :
{self.mtracks}
'''
        return _s


class BaseVisualizer():

    def __init__(self, **kwds) -> None:
        self.spb = SESSION_DATA.spb                    # steps per beat
        self.bpB = SESSION_DATA.bpB                    # beats per Bar
        self.bpM = SESSION_DATA.bpM                    # beats per Minutes
        self.tpb = SESSION_DATA.tpb                    # ticks per beat / timebase
        self.BeginTime = SESSION_DATA.BeginTime
        self.InitBar = SESSION_DATA.InitBar
        self.h = SESSION_DATA.h
        self.w = SESSION_DATA.w
        self.pitch_clip_range: list[float] = \
            SESSION_DATA.pitch_clip_range              # [A2,A7]
        self.expand_range: list[float] = \
            SESSION_DATA.expand_range                  # 音高上下拓展显示范围
        self.min_pitch_range: list[float] = \
            SESSION_DATA.min_pitch_range               # 音高上下最小范围

        for k, v in kwds.items():      # 从脚本文件中覆写参数
            if k in self.__dict__:
                self.__dict__[k] = v

        ## 计算二级参数
        self.spB = self.spb * self.bpB
        self.BpM = self.bpM / self.bpB
        self.step = self.tpb // 4      # Fl studio中可视的最小单位长度 24
        self.beat = self.step * self.spb
        self.Bar = self.beat * self.bpB

    @staticmethod
    def _init_figure(scale=1.0):
        '''
        Parameter
        ---
        scale:
            - 缩放系数，相对于2160x1080而言
        '''
        plt.close()
        fig = plt.figure(
            figsize=(6.4, 1.8),
            dpi=600 * 0.47 * scale,
            facecolor=COLOR.COLOR_BG,
            )
        ax_bg: plt.Axes = fig.add_axes([0, 0.1, 1, 0.9])
        ax_mask: plt.Axes = fig.add_axes([0, 0, 1, 0.1])
        ax_fg: plt.Axes = fig.add_axes([0.05, 0.1, 0.95, 0.9])
        ax_bg.set_xlim([0, 1])
        ax_bg.axis('off')
        [x.set_visible(False) for x in ax_mask.spines.values()]
        ax_mask.xaxis.set_visible(False)
        ax_mask.yaxis.set_visible(False)
        ax_mask.set_facecolor(COLOR.COLOR_BG)
        [x.set_visible(False) for x in ax_fg.spines.values()]
        ax_fg.yaxis.set_visible(False)
        # 绘制键盘
        ax_bg.barh(
            y=KEYBOARD.KEYBOARD_WHITE_C2E,
            width=0.05,
            height=5,
            facecolor=COLOR.COLOR_WHITE_KEY,
            edgecolor=COLOR.COLOR_EDGE,
            linewidth=0.2,
            )
        ax_bg.barh(
            y=KEYBOARD.KEYBOARD_WHITE_F2B,
            width=0.05,
            height=7,
            facecolor=COLOR.COLOR_WHITE_KEY,
            edgecolor=COLOR.COLOR_EDGE,
            linewidth=0.2,
            )
        ax_bg.barh(
            y=KEYBOARD.KEYBOARD_WHITE_C,
            width=0.05,
            height=1.5,
            facecolor=COLOR.COLOR_WHITE_C,
            edgecolor=COLOR.COLOR_EDGE,
            linewidth=0.2,
            )
        ax_bg.barh(
            y=KEYBOARD.KEYBOARD_BLACK,
            width=0.03,
            height=1,
            facecolor=COLOR.COLOR_BLACK_KEY
            )

        # 绘制背景栏
        ax_bg.barh(
            y=KEYBOARD.KEYBOARD_WHITE,
            width=0.95,
            height=1,
            left=0.05,
            facecolor=COLOR.COLOR_LIGHT_ROW,
            )
        ax_bg.barh(
            y=KEYBOARD.KEYBOARD_BLACK,
            width=0.95,
            height=1,
            left=0.05,
            facecolor=COLOR.COLOR_DARK_ROW,
            )

        # 绘制小节线
        ax_fg.xaxis.set_tick_params(
            which='both', colors=COLOR.COLOR_TICK, direction='in'
            )
        ## 设置minor=True表示设置次刻度
        ax_fg.grid(which='minor', c=COLOR.COLOR_GRID_MINOR, ls=':', lw=0.2)
        ax_fg.grid(which='major', c=COLOR.COLOR_GRID_MAJOR, ls='-', lw=0.2)

        # 设置时间线
        art_timeline, *_ = ax_fg.plot(
            [],
            [],
            color=COLOR.COLOR_TIME_LINE,
            zorder=3,
            )
        return fig, ax_bg, ax_fg, art_timeline

    def make_fig(
            self,
            song: Song,
            fig: plt.Figure,
            ax_bg: plt.Axes,
            ax_fg: plt.Axes
        ):
        '''
        给输入的midipattern渲染一张完整的静态背景

        Return
        ---
        rgba shape[h,w,4]
        '''
        # 初始化
        ax_fg._children = []
        ax_fg.containers = []
        # ax_fg.get_legend().remove()
        ax_fg.legend_ = None
        ylim = BaseVisualizer.get_disp_pitch_range(
            song.get_pitch_range(),
            self.expand_range,
            self.min_pitch_range
            )
        xlim = song.range
        ax_bg.set_ylim(ylim)
        ax_fg.set_ylim(ylim)
        ax_fg.set_xlim(xlim)
        # 虽然后面设置的小节线ticks会改变该范围，但不加xlim会导致更新不正常

        # 标记音符C
        for c in KEYBOARD.KEYBOARD_WHITE_C:
            if ylim[0] < c < ylim[1]:
                ax_bg.text(
                    x=0.048,
                    y=c,
                    s=f'C{c//12:.0f}',
                    fontsize=110 / (ylim[1] - ylim[0]),
                    va='center',
                    ha='right'
                    )

        # 小节线更新
        ax_fg.set_xticks(
            np.arange(np.floor(xlim[0]), np.ceil(xlim[1]) + 1)
            )
        ax_fg.set_xticks(
            np.arange(
                np.floor(xlim[0]), np.ceil(xlim[1]) + 1, 1 / self.bpB
                ),
            minor=True
            )

        # 绘制音符
        for chn, lbl in enumerate(song.get_instruments()):
            ax_fg.barh(
                y=song[:, lbl].track[:, 0],
                width=song[:, lbl].track[:, 4],
                height=1,
                left=song[:, lbl].track[:, 2],
                facecolor=COLOR.COLOR_NOTES_FACE[chn],
                edgecolor=COLOR.COLOR_NOTE_EDGE,
                linewidth=0.3,
                label=lbl,
                zorder=3
                )

    def display(self, song: Song):
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
        fig.canvas.draw()
        return fig

    @staticmethod
    def get_disp_pitch_range(
            pitch_range: list[float],
            expand_range: list[float] = [4, 4],
            min_pitch_range: list[float] = [10, 10]
        ):
        '''
        用于获取显示的音符范围

        Parameters
        ---
        pitch_range:
            - 原始的音高范围
        expand_range:
            - 下上边界各自需要拓宽的范围
        min_pitch_range:
            - 最小的音高范围，分下半和上半
        '''
        p_range = [
            min([
                pitch_range[0] - expand_range[0],
                round(np.mean(pitch_range)) - min_pitch_range[0]
                ]),
            max([
                pitch_range[1] + expand_range[1],
                round(np.mean(pitch_range)) + min_pitch_range[1]
                ])
            ]
        return p_range
