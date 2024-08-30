import os
import time

from dotenv import load_dotenv

load_dotenv()

DEFAULT_PAGE_SIZE = int(os.getenv("DEFAULT_PAGE_SIZE", default=5))
NEW_CLASS_PERC_TOLERANCE = float(os.getenv("NEW_CLASS_PERC_TOLERANCE", default=0.05))
NEW_EXISTING_PERC_TOLERANCE = float(
    os.getenv("NEW_EXISTING_PERC_TOLERANCE", default=0.05)
)
COS_SIM_PERC_THRESHOLD = float(os.getenv("COS_SIM_PERC_THRESHOLD", default=0.05))


def check_offset_limit(offset, limit):
    if offset is None:
        offset = 0
    if limit is None:
        limit = DEFAULT_PAGE_SIZE

    assert offset >= 0, "Offset must be a non-negative integer."
    assert limit > 0, "Limit must be a positive integer."
    assert offset % limit == 0, f"Offset must be a multiple of limit ({limit}) or 0."
    return offset, limit


def parse_arg_str(arg_str: str):
    if arg_str == "null":
        return None
    return arg_str


def get_timestamp_ms():
    return int(time.time() * 1000)
