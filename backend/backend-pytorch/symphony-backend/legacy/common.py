# some common utility functions
import collections

uid_counter = collections.defaultdict(int)

BlockId = int
NullBlockId = -1
ThreadId = int


def uid(id_type: str) -> int:
    uid_counter[id_type] += 1

    return uid_counter[id_type]


def cdiv(a, b):
    return -(a // -b)
