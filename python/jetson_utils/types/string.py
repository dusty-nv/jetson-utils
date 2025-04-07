from .table import text_table

class String:
    """
    Static class with various text parsing and formatting routines.
    """
    ParserTypes = [int, float, bool, dict, list, tuple]

    @staticmethod
    def parse(text, *default, **kwargs):
        """
        Convert the string to the given type, or best determine it as
        one of these types:  str, int, float, bool, list, dict

        The default can either be set to one of these types,
        or to a value from which the return type is inferred:
        
           my_num = String.parse("123", int)
           my_num = String.parse("456", -1)

        Upon parsing errors, either the default will be returned, 
        or the resulting exception is raised if no default was set.

        These optional arguments and kwargs are supported:

           * default (any|type): fallback value or type to return
           * type (type|str):    explicitly declare the return type
           * min (int|float):    bound the result to this minimum value
           * max (int|float):    bound the result to this maximum value
           * range (tuple|list): the (min,max) bounds as a tuple or list

        The min, max, and range kwargs are for int/float types only.
        """
        for val in default:
            if isinstance(val, type):
                kwargs.setdefault('type', val)
            elif isinstance(val, str):
                for _type in String.ParserTypes:
                    if val.lower() == _type.__name__:
                        kwargs.setdefault('type', val)
            else:
                kwargs.setdefault('default', val)

        default = kwargs.get('default')

        if 'type' not in kwargs and type(default) in String.ParserTypes:
            _type = kwargs.setdefault('type', type(default))
        else:
            _type = kwargs.get('type')

        if not text:
            if _type is None or _type == bool:
                return True  # flags with empty value
            elif 'default' not in kwargs:
                raise ValueError(f"String.parse() had empty or invalid text and no default (was type {type(text)})")
            else:
                return default

        if _type is not None:
            if _type in String.ParserTypes:
                types = [_type]
            else:
                raise ValueError(f"String.parse() had invalid type={_type} (supported types are: {[x.__name__ for x in String.ParserTypes]})")
        else:
            types = String.ParserTypes

        for _type in types:
            try:
                parser_args = kwargs.copy()
                parser_args.pop('default', None)
                parser_args.pop('type', None)
                return getattr(String, _type.__name__)(text, **parser_args)
            except Exception as error:
                last_error = error

        if 'default' in kwargs:
            return default
        else:
            raise last_error

    @staticmethod
    def int(text, *default, min=None, max=None, range=(), **kwargs):
        for val in default:
            kwargs.setdefault('default', val)

        if range and len(range) == 2:
            min, max = range

        try:
            val = kwargs.get('type', int)(text)
        except Exception as error:
            if 'default' in kwargs:
                return kwargs.get('default')
            else:
                raise error

        if min is not None and val < min:
            val = min
        if max is not None and val > max:
            val = max

        return val

    @staticmethod
    def format(text: str, length: int=None, pad: str=None, 
               html: bool=False, code: bool=False,
               sub: dict={}, **kwargs):
        """
        Apply text transformations, including padding/truncation,
        variable substitution, HTML escaping, code blocks, ect.
        """
        if not text:
            return text

        sub.update(kwargs)

        if sub:
            text = String.sub(text, **sub)

        if length:
            text = String.fit(text, length, pad)

        if html:
            text = String.html(text, code=code)
        
        return text

    @staticmethod
    def fit(text: str, length: int, pad: str=' '):
        """
        Either pad or truncate a string to get it to the desired length.
        If it is beyond the provided length, it will be truncated to fit.
        If it is shorter then it will be expanded to fill the given length
        by repeating the pad character (by default using spaces)

        TODO wrap() with ellipses like https://tutorpython.com/truncate-python-string
        """
        if not text or not length or pad is None:
            return text

        if pad == True:
            pad = ' '

        if pad and len(text) < length:
            return text + pad * (length - len(text))
                                
        if len(text) > length:
            return text[:length]
        
        return text

    @staticmethod
    def sub(text, **kwargs):
        """
        Perform variable substitution, where occurances of keys from kwargs
        are replaced in the string in the form of `$VAR` or `${VAR}`.

        The variable names are automatically capitalized, and substitution is
        applied recursively where nested references are still replaced.
        """
        if not text or not kwargs:
            return text

        while True:
            if '$' not in text: # no substitutions present
                return text

            last = text

            for k,v in kwargs.items():
                a = '$' + k.upper()
                b = '${' + k.upper() + '}'
                text = text.replace(a, b).replace(b, v)

            if text == last: # no changes made
                return text

        return text

    @staticmethod
    def html(text, code=False):
        """
        Apply escape sequences for HTML and other replacements (like '\n' -> '<br/>')
        If ``code=True``, then blocks of code will be surrounded by <code> tags.
        """
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        text = text.replace('"', '&quot;')
        text = text.replace("'", '&#039;')
        text = text.replace('\\n', '\n')
        text = text.replace('\n', '<br/>')
        text = text.replace('\\"', '\"')
        text = text.replace("\\'", "\'")
        
        if code:
            text = String.code(text)
            
        return text

    @staticmethod
    def code(text, blocks=None, open_tag='<code>', close_tag='</code>'):
        """
        Add code tags to surround blocks of code (i.e. for HTML presentation)
        Returns the text, but with the desired tags added around the code blocks.
        This works for JSON-like code with nested curly and square brackets.
        """
        if blocks is None:
            blocks = String.blocks(text)
        
        if not blocks:
            return text

        offset = 0
        
        for start, end in blocks:
            text = text[:start+offset] + open_tag + text[start+offset:end+offset] + close_tag + text[end+offset:]
            offset += len(open_tag) + len(close_tag)
            
        return text 

    @staticmethod
    def blocks(text):
        """
        Extract code blocks delimited by braces (square and curly, like JSON).
        Returns a list of (begin, end) tuples with the indices of the blocks.
        TODO - expand this to ``` delimiters, documentation sections, ect.
        """
        open_delim=0
        start=-1
        blocks=[]
        
        for i, c in enumerate(text):
            if c == '[' or c == '{':
                if open_delim == 0:
                    start = i
                open_delim += 1
            elif c == ']' or c == '}':
                open_delim -= 1
                if open_delim == 0 and start >= 0:
                    blocks.append((start,i+1))
                    start = -1
                    
        return blocks

    @staticmethod
    def table(rows, **kwargs):
        """
        This is a wrapper around the `text_table()` function and returns
        formatted/aligned text with borders to print to the terminal.
        """
        return text_table(rows, **kwargs)


__all__ = ['String']