
import tabulate
from jetson_utils import colorize

tabulate.PRESERVE_WHITESPACE = True

def text_table( rows, header=None, footer=None, 
                filter=(), color='green', 
                min_widths=[], max_widths=[30,55], 
                wrap_rows=None, merge_columns=False,
                attrs=None, **kwargs ):
    """
    Print a key-based table from a list[list] of rows/columns, or a 2-column dict 
    where the keys are column 1, and the values are column 2.  These can be wrapped
    and merged, or recursively nested like a tree of dicts.
    
    Header is a list of columns or rows that are inserted at the top.
    Footer is a list of columns or rows that are added to the end.
    
    This uses tabulate for layout, and the kwargs are passed to it:
      https://github.com/astanin/python-tabulate

    Color names and text attributes are from termcolor library:
      https://github.com/termcolor/termcolor#text-properties
    """
    if min_widths is None:
        min_widths = []

    if max_widths is None:
        max_widths = []

    kwargs.setdefault('numalign', 'center')         # set alignment kwargs to 'left', 'right', 'center'
    kwargs.setdefault('tablefmt', 'simple_outline') # set 'tablefmt' kwarg to change style

    if isinstance(rows, dict):
        rows = flatten_rows(rows, filter=filter)

    for row in rows:
        for c, col in enumerate(row):
            col = str(col)
            if c < len(min_widths) and len(col) < min_widths[c]:
                col = format_str(col, min_widths[c], pad=True)
            if c < len(max_widths) and len(col) > max_widths[c]:
                col = col[:max_widths[c]]
            row[c] = col

    if header:
        if not isinstance(header[0], list):
            header = [header]
        rows = header + rows
        
    if footer:
        if not isinstance(footer[0], list):
            footer = [footer]
        rows = rows + footer

    if wrap_rows and len(rows) > wrap_rows:
        for i in range(wrap_rows, len(rows)):
            rows[i % wrap_rows].extend(rows[i])
        rows = rows[:wrap_rows]

    if merge_columns:
        if merge_columns == True:
            merge_columns = 2
        new_columns = int(len(rows[0]) / merge_columns)
        min_widths = [0] * new_columns
        for r, row in enumerate(rows):
            for nc in range(new_columns):
                if nc*merge_columns < len(row):
                    min_widths[nc] = max(min_widths[nc], len(row[nc*merge_columns]))
        for r, row in enumerate(rows):
            new_row = []
            for nc in range(new_columns):
                if nc*merge_columns >= len(row):
                    continue

                new_col = format_str(
                    row[nc*merge_columns], 
                    length=min_widths[nc], 
                    pad=True
                )
                for x in range(1,merge_columns):
                    new_col += '  ' + row[nc*merge_columns+x]
                new_row.append(new_col)
            rows[r] = new_row

    table = tabulate.tabulate(rows, **kwargs)

    if not color:
        return table

    return '\n'.join([
        colorize(x, color, attrs=attrs)
        for x in table.split('\n')
    ])


def flatten_rows(seq, filter=()):
    """
    @internal recursively convert nested list/dict/tuple objects to a flat list.
    Children are indented so they can be printed in a simple two-column table.
    """
    def flatten(seq, indent='', prefix='', out=[]):
        iter = range(len(seq)) if isinstance(seq,(list,tuple)) else seq
        for key in iter:
            val = seq[key]
            if filter:
                val = filter(seq,key,val) 
                if isinstance(val, tuple) and len(val) == 2:
                    key, val = val
            if not val:
                continue
            if isinstance(seq,dict) and isinstance(val,list):
                flatten(val, indent, f'{key} ', out)
            elif isinstance(val, (tuple,list,dict,map)):
                out.append([indent + prefix + str(key), ''])
                flatten(val, indent + (' ├ ' if len(val) > 1 else ''), out=out)  # ┣
            else:
                out.append([indent + prefix + str(key), val])
        return out
    return flatten(seq)      


def wrap_rows(rows, max_rows=0):
    """
    @internal distribute the rows evenly across multiple columns until filled.
    """
    if not max_rows:
        return rows
    
    if len(rows) < max_rows:
        return rows

    for i in range(max_rows, len(rows)):
        rows[i % max_rows].extend(rows[i])

    return rows[:max_rows]


__all__ = ['text_table']
