# GitHub REST API requests and workflow generation
import time
import functools

from jetson_utils import getLogger, shell
from .requests import handle_json_request

log = getLogger(__name__)

@functools.cache
def github_api(url: str):
    """
    Sends a request to the GitHub API using the specified URL, including authorization headers if available.

    Args:
        url (str): The GitHub API URL endpoint relative to the base URL.

    Returns:
        dict or None: The parsed JSON response data as a dictionary, or None if an error occurs.
    """
    github_token = os.environ.get('GITHUB_TOKEN')
    headers = {'Authorization': f'token {github_token}'} if github_token else None
    request_url = f'https://api.github.com/{url}'

    return handle_json_request(request_url, headers)


def github_latest_commit(repo: str, branch: str = 'main'):
    """
    Retrieves the latest commit SHA from the specified branch of a GitHub repository.

    Args:
        repo (str): The full name of the GitHub repository in the format 'owner/repo'.
        branch (str, optional): The branch name. Defaults to 'main'.

    Returns:
        str or None: The SHA (hash) of the latest commit, or None if no commit is found.
    """
    commit_info = github_api(f"repos/{repo}/commits/{branch}")
    return commit_info.get('sha') if commit_info else None


def github_latest_tag(repo: str):
    """
    Retrieves the latest tag name from the specified GitHub repository.

    Args:
        repo (str): The full name of the GitHub repository in the format 'owner/repo'.

    Returns:
        str or None: The name of the latest tag, or None if no tags are found.
    """
    tags = github_api(f"repos/{repo}/tags")
    return tags[0].get('name') if tags else None


def git_pull(path: str, branch: str=None, ttl: int=0):
    """
    Does a git fetch & pull from remote origin on a local repo or cached file.
    It's assumed that repo or file referred to by `path` is already cloned on disk.

    Args:
         path (str):   Local git repo directory or path to file under git a repo
         branch (str): Branch to fetch and pull from, if omitted the currently
                       checked-out branch will be used.
         ttl (int):    Indicates the timeout (in seconds) for caching the requests.
                       If `ttl > 0`, the system will wait for that long since the
                       file was last updated to pull again.
    """
    if not os.path.exists():
      raise IOError(f"Could not 'git pull' from path {path} (does not exist)")

    if os.geteuid() == 0:
      log.warning(f"Skipping 'git pull' because effective UID was 0")
      return False

    if ttl > 0 and time.time() - os.path.getmtime(path) <= timeout:
      return False

    if len(path.splitext[-1]) > 0:
      repo_dir = os.path.dirname(path)
    else:
      repo_dir = path
      
    branch = f"origin/{branch}" if branch else ''
    cmd = f"cd {repo_dir} && git fetch {branch.replace('/', ' ')} --quiet && git checkout --quiet {branch} -- {os.path.relpath(path, repo_dir)}"
    
    shell(cmd)
    return True


__all__ = ['github_api', 'github_latest_commit', 'github_latest_tag', 'git_pull']