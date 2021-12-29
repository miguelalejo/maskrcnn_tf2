import sys
import time

sys.path.append('../src')
import coco_minitrain
from typing import Callable


def timing(func: Callable):
    """
    Time measurement decorator
    Args:
        func: training, Callable
    Returns: Callable
    """

    def measure(*args):
        start_time = time.perf_counter()
        func(*args)
        print("\nExecution time:", time.perf_counter() - start_time)

    return measure


@timing
def test_coco_mini() -> None:
    args = coco_minitrain.parse_arguments()
    coco_minitrain.coco_mini(args)


if __name__ == '__main__':
    sys.exit(test_coco_mini())
