from enum import Enum

class AggregationsTypes(Enum):
    MEAN = 1
    MEDIAN = 2
    MAX = 3
    MIN = 4
    COUNT = 5
    NUNIQUE = 6
    SUM = 7
    MODE = 8