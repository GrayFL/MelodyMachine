from enum import Enum

# Sesstion_Data = {
#     'StartBar': 5,  # .flp工程中的音频实际起始小节号
#     'InitBar': 1,  # 视频中的起始记数小节号
#     'BeginTime': 1,  # 开幕大标题的显示时长（秒）
#     'CountDown': 3,  # 倒计时个数
#     'Title': '',  # 开幕大标题文本
#     'Saying': '',  # 开幕格言文本
#     'Name': '—— Gray Frezicical',  # 落款
#     'h': 1080,  # 视频的高度（像素）
#     'w': 2160,  # 视频的宽度（像素）
#     'FontPath':  # 字体路径  \
#     'C:/Users/Gray/AppData/Local/Microsoft/Windows/Fonts/sarasa-mono-sc-regular.ttf',
#     'spb': 4,  # steps per beat
#     'bpB': 4,  # beats per Bar
#     'bpM': 120,  # beats per Minutes
#     'tpb': 96,  # ticks per beat / timebase
#     'pitch_clip_range': [33, 93],  # 全局音符音高范围[A2,A7]
#     'expand_range': [4, 4],  # 音高上下拓展显示范围
#     'min_pitch_range': [10, 10],  # 音高上下最小范围
#     'subclip_tBar': None,  # 输出视频的裁剪范围（小节）
#     'subclip_tMov': None,  # 输出视频的裁剪范围（秒）
#     }


class SESSION_DATA():
    StartBar = 5  # .flp工程中的音频实际起始小节号
    InitBar = 1  # 视频中的起始记数小节号
    BeginTime = 1  # 开幕大标题的显示时长（秒）
    CountDown = 3  # 倒计时个数
    Title = ''  # 开幕大标题文本
    Saying = ''  # 开幕格言文本
    Name = '—— Gray Frezicical',  # 落款
    h = 1080  # 视频的高度（像素）
    w = 2160  # 视频的宽度（像素）
    FontPath = 'C:/Users/Gray/AppData/Local/Microsoft/Windows/Fonts/sarasa-mono-sc-regular.ttf'  # 字体路径
    spb = 4  # steps per beat
    bpB = 4  # beats per Bar
    bpM = 120  # beats per Minutes
    tpb = 96  # ticks per beat / timebase
    pitch_clip_range = [33, 93]  # 全局音符音高范围[A2,A7]
    expand_range = [4, 4]  # 音高上下拓展显示范围
    min_pitch_range = [10, 10]  # 音高上下最小范围
    subclip_tBar = None  # 输出视频的裁剪范围（小节）
    subclip_tMov = None  # 输出视频的裁剪范围（秒）
    spB = spb * bpB
    BpM = bpM / bpB
    step = tpb // 4  # Fl studio中可视的最小单位长度 24
    beat = step * spb
    Bar = beat * bpB


if '__main__' == __name__:
    pass
