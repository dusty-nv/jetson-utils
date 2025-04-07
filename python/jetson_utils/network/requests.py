import os
import requests, time
import functools

from typing import Dict, Literal
from jetson_utils import getLogger

log = getLogger(__name__)

def handle_text_request(url, retries=3, backoff=5):
    """
    Handles a request to fetch text data from the given URL.

    Args:
        url (str): The URL from which to fetch text data.

    Returns:
        str or None: The fetched text data, stripped of leading and trailing whitespace,
                     or None if an error occurs.
    """
    for attempt in range(retries):
        try:
            log.verbose(f"Fetching text  {url} (attempt {attempt+1})")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.text.strip()
        except Exception as e:
            log.warning(f"Failed to fetch text from {url}: {e}")
            if attempt < retries - 1:
                time.sleep(backoff)
            else:
                return None


def handle_json_request(url: str, headers: Dict[str, str] = None, 
                        retries: int = 3, backoff: int = 3, timeout: int = 10):
    """
    Fetch JSON data from a URL with retry, timeout, and backoff handling.

    Args:
        url (str): The URL to fetch.
        headers (dict): Optional HTTP headers.
        retries (int): Number of retry attempts.
        backoff (int): Seconds to wait between retries.
        timeout (int): Timeout in seconds for each request.

    Returns:
        dict or None: Parsed JSON response, or None if all attempts fail.
    """
    for attempt in range(1, retries + 1):
        try:
            log.verbose(f"Fetching json  {url} (attempt {attempt})")
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            log.error(f"HTTP {e.response.status_code} while fetching {url}")
        except requests.RequestException as e:
            log.warning(f"Request error on {url}: {e}")
        except Exception as e:
            log.warning(f"Unexpected error on {url}: {e}")

        if attempt < retries:
            time.sleep(backoff)
        else:
            log.error(f"All attempts to fetch {url} failed.")

    return None


def get_json_value_from_url(url: str, notation: str = None):
    """
    Retrieves JSON data from the given URL and returns either the whole data or a specified nested value using a dot notation string.

    Args:
        url (str): The URL from which to fetch the JSON data.
        notation (str, optional): A dot notation string specifying the nested property to retrieve.
                                  If None or an empty string is provided, the entire JSON data is returned.

    Returns:
        str or dict: The value of the specified nested property or the whole data if `notation` is None.
                     Returns None if the specified property does not exist.
    """
    data = handle_json_request(url)

    if notation and data is not None:
        keys = notation.split('.') if '.' in notation else [notation]
        current = data

        try:
            for key in keys:
                current = current[key]
            return str(current).strip()
        except KeyError as e:
            log.error(f'Failed to get the value for {notation}: {e}')
            return None

    return data
