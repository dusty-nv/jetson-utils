import copy
import pprint

from .string import String

__all__ = ['NamedDict', 'merge_dicts', 'filter_keys']

class NamedDict(dict):
    """
    A dict where keys are available as named attributes:
    
      https://stackoverflow.com/a/14620633
      
    So you can do things like:
    
      x = NamedDict(a=1, b=2, c=3)
      x.d = x.c - x['b']
      x['e'] = 'abc'
      
    This is using the __getattr__ / __setattr__ implementation
    because of memory leaks encountered without it:
    
      https://bugs.python.org/issue1469629
    """
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    def __getattr__(self, key):
        if key and key.startswith('__'):
            return super().__getattr__(key)
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, value):
        self.__dict__ = value

    def __str__(self):
        try:
            return pprint.pformat(self, indent=2)
        except Exception as error:
            return super().__str__()
        
    def deepcopy(self):
        return copy.deepcopy(self)

    def filter(self, **kwargs):
        return filter_keys(self, **kwargs)

    def parse(self, **kwargs):
        for k,v in self.items():
            if isinstance(v, str):
                self[k] = String.parse(v, default=v, **kwargs)
        return self


def merge_dicts(src, dst, recursive=True, replace=True):
    """
    Recursively merge the source dictionary into the destination.
    """
    for key, value in src.items():
        if isinstance(value, dict):
            node = dst.setdefault(key, {})
            if recursive:
                merge_dicts(value, node)
        else:
            if not replace or key not in dst:
                dst[key] = copy.deepcopy(value)

    return dst


def filter_keys(src, dst=None, include=[], exclude=[], inplace=False):
    """
    Remove keys from a dict by either a list of keys to keep or remove.
    """
    if inplace:
        if dst:
            raise ValueError(f"filter_keys() should only have one of 'inplace' or 'dst' set")
        else:
            dst = src
    else:
        if dst is None:
            dst = src.__class__()
            
    if isinstance(src, list):
        for idx, obj in enumerate(src):
            res = filter_keys(obj, include=include, exclude=exclude, inplace=inplace)
            if idx < len(dst):
                dst[idx] = res
            else:
                dst.append(res)
        return dst
        
    for key in list(src.keys()):
        if (include and key not in include) or (exclude and key in exclude):
            if inplace:
                del src[key]
        elif not inplace:
            dst[key] = src[key]

    return dst
