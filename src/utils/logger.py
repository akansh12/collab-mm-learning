import logging

logger = logging.getLogger("mmfl")
formatter = logging.Formatter(
    fmt="%(filename)s | %(lineno)d | %(funcName)s | %(asctime)s | %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger.setLevel(logging.INFO)

_handler = logging.StreamHandler()
_handler.setLevel(logging.INFO)
_handler.setFormatter(formatter)
logger.addHandler(_handler)
