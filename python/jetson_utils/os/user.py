import os
import grp


def user_in_group(group) -> bool:
    """
    Returns true if the user running the current process is in the specified user group.
    Equivalent to this bash command:   id -nGz "$USER" | grep -qzxF "$GROUP"
    """
    try:
        group = grp.getgrnam(group)
    except KeyError:
        return False

    return (group.gr_gid in os.getgroups())


def is_root_user() -> bool:
    """
    Returns true if this is the root user running
    """
    return os.geteuid() == 0


def needs_sudo(group: str = 'docker') -> bool:
    """
    Returns true if sudo is needed to use the docker engine (if user isn't in the docker group)
    """
    if is_root_user():
        return False
    else:
        return not user_in_group(group)


def sudo_prefix(group: str = 'docker'):
    """
    Returns a sudo prefix for command strings if the user needs sudo for accessing docker
    """
    if needs_sudo(group):
        return "sudo "
    else:
        return ""


__all__ = ['user_in_group', 'is_root_user', 'needs_sudo', 'sudo_prefix']