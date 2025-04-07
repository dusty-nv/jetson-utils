import os
import termcolor


def colorize(text, color=None, on_color=None, attrs=[]):
    """
    Apply ANSI terminal color codes - supports some inline tags like `<b>Bold</b>`
    """
    if not text:
        return None

    if 'ANSI_COLORS_DISABLED' in os.environ:
        return text.replace('<b>', '').replace('</b>', '')

    if attrs and isinstance(attrs, str):
        attrs = [attrs]

    # https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797#colors--graphics-mode
    text = text.replace('<b>', f'\033[1m').replace('</b>', f'\033[22m')

    if color or on_color or attrs:
        text = termcolor.colored(text, color=color, on_color=on_color, attrs=attrs)
    
    return text


def cprint(text, color=None, on_color=None, attrs=[], **kwargs):
    """
    Print string to terminal in the specified color (from termcolor)
    """
    kwargs.setdefault('flush', True)
    print(colorize(text, color, on_color, attrs), **kwargs)


__all__ = ['colorize', 'cprint']