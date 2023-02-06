def loadList(list_name, list_path=''):
    """Load a file list."""

    with open(os.path.join(list_path, f'{list_name}.list'), 'r') as f:
        file_list = [line.strip() for line in f.readlines()]
    
    return file_list


def loadLists(list_names, list_path=''):
    """Load file lists."""
    
    list_dict = dict()
    for list_name in list_names:
        list_dict[list_name] = loadList(list_name, list_path=list_path)
    
    return list_dict