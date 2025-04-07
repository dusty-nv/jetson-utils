# Some basic utilities for finding/starting/stopping containers
#
# These require 'pip install docker' and when in another container,
# for the docker daemon's socket to be mounted like so: 
#
#   -v /var/run/docker.sock:/var/run/docker.sock
#
# It will then have access to the container state of the host device,
# and is not using true 'docker-in-docker' with an independent daemon.
import docker
import logging

log = logging.getLogger()

class Docker:
    """
    Some basic utilities for starting/stopping containers
    This requires the docker socket to be mounted:
      /var/run/docker.sock:/var/run/docker.sock
    """
    Client = None

    @staticmethod
    def client():
        if not Docker.Client:
            Docker.Client = docker.from_env()
        return Docker.Client
    
    @staticmethod
    def find(names):
        if isinstance(names, str):
            names=[names]
        try:
            for c in Docker.client().containers.list():
                for name in names:
                    if name.lower().replace('_','-') in c.name.lower().replace('_','-'):
                        return c.name
            log.warning(f"Failed to find container by the names {names}")
        except Exception as error:
            log.error(f"Exception trying to find container {names} ({error})")

    @staticmethod
    def stop(name, remove=True):
        try:
            name = Docker.find(name)
            if name:
                c = Docker.client().containers.get(name)
                log.info(f"Stopping container '{c.name}' ({c.id})")
                c.stop()
        except Exception as error:
            log.error(f"Failed to stop container '{name}' ({error})")
            Docker.kill(name)
        if remove:
            Docker.remove(name)

    @staticmethod
    def kill(name):
        name = Docker.find(name)
        if name:
            c = Docker.client().containers.get(name)
            log.info(f"Killing container '{c.name}' ({c.id})")
            c.kill()

    @staticmethod
    def remove(name, force=True):
        try:
            name = Docker.find(name)
            if not name:
                return
            c = Docker.client().containers.get(name)
            log.info(f"Removing container '{c.name}' ({c.id})")
            c.remove(force=force)
        except Exception as error:
            log.error(f"Failed to remove container '{name}' ({error})")

__all__ = ['Docker']