import numpy as np
import matplotlib.pyplot as plt


class ColorPalette():

    def __init__(self, list_of_colors: np.ndarray | list) -> None:
        self.colors = list_of_colors
        self.len = len(list_of_colors)

    def __getitem__(self, idx):
        return self.colors[idx % self.len]


class COLOR():
    # 颜色
    COLOR_BG = np.array([49 / 255, 56 / 255, 62 / 255])
    COLOR_BLACK_KEY = np.array([0.23, 0.23, 0.23])
    COLOR_WHITE_KEY = np.array([0.9, 0.9, 0.9])
    COLOR_WHITE_C = np.array([0.75, 0.75, 0.75])
    COLOR_LIGHT_ROW = np.array([0.5, 0.5, 0.5, 0.1])
    COLOR_DARK_ROW = np.array([0.2, 0.2, 0.2, 0.0])
    COLOR_EDGE = np.array([0.5, 0.5, 0.5])
    COLOR_NOTE_FACE = np.array([0.9, 0.9, 0.9])
    # COLOR_NOTES_FACE = ['#9ed1a5', '#9fd3ba', '#a1d6d0', '#a3cad8']
    # COLOR_NOTES_FACE = [plt.get_cmap('tab20')(x) for x in range(20)]
    COLOR_NOTES_FACE = ColorPalette([
        plt.get_cmap('tab20')(x) for x in [5, 1, 3, 9, 17, 15]
        ])
    # COLOR_NOTES_FACE = [plt.get_cmap('tab20')(x) for x in [4, 0, 2, 8, 16, 14]]
    COLOR_NOTE_EDGE = np.array([*COLOR_BG, 0.9])
    COLOR_TICK = np.array([0.9, 0.9, 0.9])
    COLOR_GRID_MAJOR = np.array([0.8, 0.8, 0.8])
    COLOR_GRID_MINOR = np.array([0.6, 0.6, 0.6])
    COLOR_TIME_LINE = np.array([0.1, 0.5, 0.7])


class KEYBOARD():
    KEYBOARD_BLACK = (
        np.tile(np.arange(2, 9) * 12,
                (5, 1)) + np.array([[1, 3, 6, 8, 10]]).T
        ).flatten('F')
    KEYBOARD_WHITE = (
        np.tile(np.arange(2, 9) * 12,
                (7, 1)) + np.array([[0, 2, 4, 5, 7, 9, 11]]).T
        ).flatten('F')
    KEYBOARD_WHITE_C2E = (
        np.tile(np.arange(2, 9) * 12, (1, 1)) + np.array([[2]]).T
        ).flatten('F')
    KEYBOARD_WHITE_F2B = (
        np.tile(np.arange(2, 9) * 12, (1, 1)) + np.array([[8]]).T
        ).flatten('F')
    KEYBOARD_WHITE_C = (
        np.tile(np.arange(2, 9) * 12, (1, 1)) + np.array([[0.25]]).T
        ).flatten('F')
