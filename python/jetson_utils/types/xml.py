import xml.etree.ElementTree as ET


def xmlToJson(tree, nan=[], blacklist=[], rename={}):
    """ 
    Convert XML to JSON and filter the keys.
    (this gets used to parse nvidia-smi output)
    """
    response = {}

    if not nan:
        nan = ['N/A', 'Unknown Error', 'None', None]

    if not blacklist:
        blacklist = [
            'gpu_reset_status', 'ibmnpu', 'temperature',
            'gpu_power_readings', 'module_power_readings'
        ]

    if not rename:
        rename = {
            'product_name': 'name',
            'product_architecture': 'arch',
        }

    def is_nan(text):
        text = text.lower()
        for n in nan:
            if n:
                if n.lower() in text:
                    return True
            else:
                if not text:
                    return True
        return False
    
    if isinstance(tree, str):
        tree = ET.fromstring(tree)

    for child in tree:
        if child.tag in blacklist:
            continue
        if child.tag in rename:
            child.tag = rename[child.tag]

        if len(list(child)) > 0:
            children = xmlToJson(child)
            if children:
                if child.tag in response:
                    if isinstance(response[child.tag], list):
                        response[child.tag].append(children)
                    else:
                        response[child.tag] = [response[child.tag], children]
                else:      
                    response[child.tag] = children
        else:
            text = child.text.strip()
            if not is_nan(text):
                response[child.tag] = text

    return response


__all__ = ['xmlToJson']
