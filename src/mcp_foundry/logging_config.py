import logging
import sys
import io

def configure_utf8_logging():
    # Ensure UTF-8 logger output on all platforms
    # Use stderr because MCP protocol mandates that stdout is used for data following pure JSON-RPC over stdout/stdio.
    utf8_stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    handler = logging.StreamHandler(utf8_stderr)
    # Message will have logging level information, although it will be passed to stderr stream.
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)