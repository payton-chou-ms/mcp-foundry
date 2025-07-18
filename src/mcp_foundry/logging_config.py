import logging
import sys
import io

# Store the wrapped stderr stream to avoid multiple wrappers
_utf8_stderr = None

def configure_utf8_logging():
    global _utf8_stderr
    
    # Ensure UTF-8 logger output on all platforms
    # Use stderr because MCP protocol mandates that stdout is used for data following pure JSON-RPC over stdout/stdio.
    if _utf8_stderr is None:
        _utf8_stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    handler = logging.StreamHandler(_utf8_stderr)
    
    # Message will not have timestamp because MCP Host will add the timestamp.
    # Message will have logging level information, although it will be passed to stderr stream.
    formatter = logging.Formatter(
        fmt='[%(levelname)-8s] [%(name)s] %(message)s',
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)