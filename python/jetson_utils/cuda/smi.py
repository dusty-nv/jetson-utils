import ctypes
import logging

from jetson_utils import xmlToJson


log = logging.getLogger(__name__)


def nvidia_smi_query():
    """ 
    Query GPU device info from nvidia-smi.
    """
    try:
        return xmlToJson(nim.subshell('nvidia-smi -q -x', echo=False, dry_run=False))
    except Exception as error:
        log.warning(f'Failed to query GPU devices from nvidia-smi ({error})')
        return {}


__all__ = ['nvidia_smi_query']
