import re
import logging
from pathlib import Path
from typing import Optional


def get_logger(
    name: Optional[str] = 'main',
    level: int = logging.INFO,
    log_to_console: bool = False,
    console_format: Optional[str] = None,
    console_level: Optional[int] = None,
    log_to_file: bool = False,
    filename: Optional[str] = None,
    file_format: Optional[str] = None,
    file_level: Optional[str] = None,
) -> logging.Logger:
    """
    Custom logger.

    If logger called `name`, at the time this function is called, already has 
    a `StreamHandler` and/or `MyTimedRotatingFileHandler`, then no new 
    `StreamHandler` and/or `MyTimedRotatingFileHandler` will be added.

    That is, for example:
    >>> logger = get_logger(name='my_logger_name', log_to_file=False)

    will return `'my_logger_name'` with `StreamHandler` only (since 
    `log_to_file=False`). But calling it again as so:

    >>> logger = get_logger(name='my_logger_name', log_to_file=True)

    will return `'my_logger_name'` with both `StreamHandler` and
    `FileHandler`. Since loggers are singletons, the second
    call will modify the logger in all other modules where it is used.

    Parameters
    ----------
    name: str | None
        The  name of the logger. Once a logger is created, given the same
        name the same logger will be always returned. If `None`, the root logger
        will be returned (not advised). A child logger can be created by 
        prefixing the child logger name with the parent logger name separated 
        by a dot; e.g., `'A.B.C'` is a child of logger `'A.B'` which in turn is 
        a child of `'A'`. Default is `'main'`.
    level: int
        An integer value that defines the main logging level of the whole logger.
        Default is `logging.INFO` (=20).
    log_to_console: bool
        If `True`, a stream handler will be added to the logger to output logs
        to the console. If `False`, `console_format` and `console_level` will be
        ignored. Default is `False`.
    console_format: str | None
        Format string that will be passed to `logging.Formatter` added to the 
        stream handler. If `None`, default format string will be used. 
    console_level: str | None
        An integer value that defines the stream handler's logging level. If 
        `None`, the main logging level, `level`, will be used.
    log_to_file: bool
        If `True`, a file handler will be added to the logger to output logs
        to a file. If `False`, `filename`, `file_format`, and `file_level` 
        will be ignored. Default is `False`.
    filename: str | None
        Base part of logs filename. If `None`, the name of the directory will
        be used.
    file_format: str | None
        Format string that will be passed to `logging.Formatter` added to the 
        file handler. If `None`, default format string will be used. 
    file_level: str | None
        An integer value that defines the file handler's logging level. If 
        `None`, the main logging level, `level`, will be used.

    Returns
    -------
    logging.Logger 
        The custom logger instance.

    Example
    -------
    In main file:
    >>> logger = get_logger(name='my_logger_name', ...)
    In other files:
    >>> import logging
    >>> logger = logging.getLogger('my_logger_name')
    """
    # create the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(
        '[%(asctime)s] {%(module)s:%(funcName)s:%(lineno)d} (%(name)s) %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    has_stream_handler = any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers)
    has_file_handler = any(isinstance(handler, logging.FileHandler) for handler in logger.handlers)
    # create stream handler
    if not has_stream_handler and log_to_console:
        stream_handler = logging.StreamHandler()
        console_formatter = formatter
        if console_format is not None:
            console_formatter = logging.Formatter(console_format)
        stream_handler.setFormatter(console_formatter)
        if console_level is not None:
            stream_handler.setLevel(console_level)
        logger.addHandler(stream_handler)
    # create file handler
    if not has_file_handler and log_to_file:
        # create the log filename; if exists add an integer to the end of stem.
        if filename is None:
            filename = 'logs.log'
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        i = 1
        while filename.exists():
            pattern = re.compile(r'\.\d+\.log')
            if pattern.findall(str(filename)):
                filename = Path(pattern.sub(f'.{i}.log', str(filename)))
            else:
                filename = filename.with_name(filename.stem+f'.{i}.log')
            i += 1

        file_handler = logging.FileHandler(filename)
        file_formatter = formatter
        if file_format is not None:
            file_formatter = logging.Formatter(file_format)
        file_handler.setFormatter(file_formatter)
        if file_level is not None:
            file_handler.setLevel(file_level)
        logger.addHandler(file_handler)
    return logger


class LoggerWriter:

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.buffer = []

    def write(self, msg: str) -> None:
        if msg.endswith('\n'):
            self.buffer.append(msg.rstrip('\n'))
            self.logger(''.join(self.buffer))
            self.buffer = []
        else:
            self.buffer.append(msg)

    def flush(self) -> None:
        pass

