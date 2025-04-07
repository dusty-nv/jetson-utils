# this module adds extensions to the standard python Logging interface,
# including a custom Logger class with methods like .success() and .status()
import logging

from .terminal import colorize, cprint 

# add custom logging levels
logging.SUCCESS = 35 
logging.VERBOSE = 15 

__all__ = [
    'Logger', 'LogFormatter', 'basicConfig',  
    'getLogger','logSuccess', 'logVerbose'
]

class Logger(logging.Logger):
    """
    Customized Logger created by default after `logging.setLoggerClass()` is called below.
    """
    DefaultLevel = logging.DEBUG

    def __init__(self, name: str) -> None:
        super().__init__(name)
        
        handler = logging.StreamHandler()
        handler.setFormatter(LogFormatter())

        self.addHandler(handler)
        self.setLevel(Logger.DefaultLevel)

    def success(self, *args, **kwargs):
        self.log(logging.SUCCESS, *args, **kwargs)

    def verbose(self, *args, **kwargs):
        self.log(logging.VERBOSE, *args, **kwargs)

    def getLevels(self):
        return dict(
            debug = logging.DEBUG,
            verbose = logging.VERBOSE,
            info = logging.INFO,
            success = logging.SUCCESS,
            warning = logging.WARNING,
            error = logging.ERROR,
            critical = logging.CRITICAL
        )

class LogFormatter(logging.Formatter):
    """
    Colorized log formatter (inspired from https://stackoverflow.com/a/56944256)
    Use basicConfig() instead to enable it with the desired logging level.
    """  
    DefaultFormat = '[%(asctime)s] %(message)s' #'%(asctime)s | %(levelname)s | %(message)s'
    DefaultDateFmt = '%H:%M:%S'
    
    DefaultColors = {
        logging.DEBUG: (None, 'dark'), # 'light_grey'
        logging.VERBOSE: (None, 'dark'),
        logging.INFO: (None, 'dark'),
        logging.SUCCESS: 'green',
        logging.WARNING: 'yellow',
        logging.ERROR: 'red',
        logging.CRITICAL: 'red'
    }

    def __init__(self, format: str=None, datefmt: str=None, colors: dict=None, **kwargs):
        """ @internal Use basicConfig() instead of instantiating this directly. """
        self.formatters = {}
        
        format = format if format else LogFormatter.DefaultFormat
        datefmt = datefmt if datefmt else LogFormatter.DefaultDateFmt
        colors = colors if colors else LogFormatter.DefaultColors

        for level in LogFormatter.DefaultColors:
            if level in colors and colors[level] is not None:
                color = colors[level]
                attrs = None
                
                if not isinstance(color, str):
                    attrs = color[1:]
                    color = color[0]

                fmt = colorize(format, color, attrs=attrs)
            else:
                fmt = format
                
            self.formatters[level] = logging.Formatter(fmt=fmt, datefmt=datefmt)

    def format(self, record):
        """ Implementation of logging.Formatter record formatting function """
        return self.formatters[record.levelno].format(record)


def basicConfig(level: str='info', format: str=None, datefmt: str=None, colors: dict=None, **kwargs):
    """
    Configure the root logger with formatting and color settings.
    
    Parameters:
        level (str|int) -- Either the log level name 
        format (str) -- Message formatting attributes (https://docs.python.org/3/library/logging.html#logrecord-attributes)
        
        datefmt (str) -- Date/time formatting string (https://docs.python.org/3/library/logging.html#logging.Formatter.formatTime)
        
        colors (dict) -- A dict with keys for each logging level that specify the color name to use for those messages
                        You can also specify a tuple for each couple, where the first entry is the color name,
                        followed by style attributes (from https://github.com/termcolor/termcolor#text-properties)
                        If colors is None, then colorization will be disabled in the log.
                        
        kwargs (dict) -- Additional arguments passed to logging.basicConfig() (https://docs.python.org/3/library/logging.html#logging.basicConfig)
    """
    logging.addLevelName(logging.STATUS, 'STATUS')
    logging.addLevelName(logging.SUCCESS, 'SUCCESS')

    logging.success = logSuccess
    
    if not level:
        level = DefaultLevel

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(LogFormatter(format=format, datefmt=datefmt, colors=colors))

    #handler.setLevel(level)
    #if len(logging.getLogger().handlers) > 0:
    #    logging.getLogger().removeHandler(logging.getLogger().handlers[0])
    #logger.handlers.clear()
    #logging.getLogger(__name__).setLevel(level)
    
    logging.basicConfig(handlers=[handler], level=level, force=True, **kwargs)


def getLogger(name=__name__):
    """ Shortcut for importing logging and `logger.getLogger(__name__)` """
    logger = logging.getLogger(name=name)
    logger.basicConfig = basicConfig
    return logger

def logSuccess(*args, **kwargs):
    """ Global for custom logging.SUCCESS level """
    getLogger().log(logging.SUCCESS, *args, **kwargs)

def logVerbose(*args, **kwargs):
    """ Global for custom logging.VERBOSE level """
    getLogger().log(logging.VERBOSE, *args, **kwargs)


# enable this as the default logger type
logging.setLoggerClass(Logger)
