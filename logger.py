import logging

TRACE = 5
logging.addLevelName(TRACE, "TRACE")

logger = logging.getLogger("proxy")


def log_trace(msg, *args, context=None):
    prefix = f"{context} " if context else ""
    logger.log(TRACE, prefix + msg, *args)


def log_info(msg, *args, context=None):
    prefix = f"{context} " if context else ""
    logger.info(prefix + msg, *args)


def log_debug(msg, *args, context=None):
    prefix = f"{context} " if context else ""
    logger.debug(prefix + msg, *args)


def log_warn(msg, *args, context=None):
    prefix = f"{context} " if context else ""
    logger.warning(prefix + msg, *args)
