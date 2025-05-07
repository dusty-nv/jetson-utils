# In jetson-utils v1, all python functionality was through C/C++ bindings (under python/bindings)
# In jetson-utils v2, native python modules were added under `python/jetson_containers`
#
# This C extension module (jetson_utils_python) is required for jetson-utils v1, but optional for v2.
# Also in v2 the VERSION/__version__ define is pulled from the packaging system instead of hardcoded.
try:
    import importlib.metadata
    __version__ = importlib.metadata.version('jetson_utils')
except Exception as error:
    __version__ = '1.0.0'
finally:
    VERSION = __version__  # backwards compatibility with v1

try:
    from jetson_utils_python import *
    HAS_JETSON_UTILS_EXT = True
except Exception as error:
    HAS_JETSON_UTILS_EXT = False
    if __version__.startswith('1.'):  # C extension required for v1
        raise error

from .os import *
from .types import *

from .network import *
from .cuda import *