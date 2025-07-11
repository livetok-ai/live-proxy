import logging

logger = logging.getLogger("proxy")

def log_info(msg, *args, context=None):
    prefix = f"{context} " if context else ""
    logger.info(prefix + msg, *args) 