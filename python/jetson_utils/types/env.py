import os

from .dict import NamedDict
from .string import String

# Parse environment from OS
Env = NamedDict(**os.environ).parse().filter(
    exclude=[ # leave out some long/unused ones
        'DBUS_SESSION_BUS_ADDRESS',
        'DEBUINFOD_URLS',
        'LESSCLOSE',
        'LESSOPEN',
        'LOGNAME',
        'LS_COLORS',
        'PS1',
    ]
)

def getenv(keys: list[str], default=None, dtype=None, **kwargs):
    """
    This getenv() extends the Python version with type casting, empty flags,
    and multiple keys where it's validated if any of those keys are defined.

    If keys is a list or tuple, then the first present key will be used.
    For example, this will return `$CACHE_DIR` first and so forth:

      getenv(('CACHE_DIR', 'HF_HOME', 'XGD_HOME'), '/root/.cache')

    If none of those keys are found, then the default value is returned.
    If dtype is specified, then String.parse() is called with kwargs first.
    """
    if not keys:
        return default

    if isinstance(keys, str):
        keys = [keys]

    keys = [k for k in keys if k in Env]

    if len(keys) == 0:
        return default

    if dtype is None or type(Env[k]) == dtype:
        return Env[k]
    elif k in os.environ:
        return String.parse(os.environ[k], default=default, dtype=dtype, **kwargs)       


__all__ = ['Env', 'getenv']