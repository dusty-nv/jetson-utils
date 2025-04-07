import shutil
import subprocess

from .logging import colorize, getLogger

log = getLogger()

def shell(cmd, echo=True, capture_output=False, dry_run=None, **kwargs):
    """ 
    Run shell command and return the result 
    """
    if dry_run is None:
        dry_run = os.environ.get('DRY_RUN', None)

    if not isinstance(cmd, list):
        cmd = [cmd]
        
    cmd = [x for x in cmd if x != None and len(x) > 0]

    if echo:
        endline = f' \\\n    '
        
        dry_run_msg = '(Skipping shell command during DRY RUN)'
        default_msg = dry_run_msg if dry_run else 'Running shell command'

        if isinstance(echo, str):
            echo = echo + ' ' + (dry_run_msg if dry_run else '') 
        else:
            echo = default_msg

        log.info(f"{echo}\n\n  {endline.join(cmd)}\n")

    kwargs.setdefault('executable', '/bin/bash')
    kwargs.setdefault('shell', True)
    kwargs.setdefault('check', True)
    kwargs.setdefault('capture_output', capture_output)
    kwargs.setdefault('text', capture_output)

    return subprocess.run('' if dry_run else ' '.join(cmd), **kwargs)


def subshell(cmd, capture_output=True, **kwargs):
    """ 
    Run a shell and capture the output by default
    """
    return shell(cmd, capture_output=capture_output, **kwargs).stdout


def has_command(exe):
    """
    Return true if there's an executable found in the PATH by this name. 
    """
    return shutil.which(exe) is not None


def try_import(module):
    """ 
    Return true if import succeeds, false otherwise 
    """
    try:
        __import__(module)
        return True
    except ImportError as error:
        log.debug(f"{module} not found ({error})")
        return False


__all__ = ['shell', 'subshell', 'has_command', 'try_import']