import logging
import sys
import io

def configure_utf8_logging():
    # Ensure UTF-8 logger output on all platforms
    utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    handler = logging.StreamHandler(utf8_stdout)
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)