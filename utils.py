import enum

VERBOSE_LEVEL = 3


class Level(enum.IntEnum):
    WARNING = 1
    INFO = 2
    DEBUG = 3


def print_d(str, level):
    if level <= VERBOSE_LEVEL:
        print(str)
