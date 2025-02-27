from typing import Union
from numpy import str_, number, dtype, integer

AllStrType = Union[str, str_]
AllNumType = Union[int, float, number]
AllIntType = Union[int, integer]
NotationType = dtype('U4')
PitchType = dtype('int8')
