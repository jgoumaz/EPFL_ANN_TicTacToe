

def index_to_position(index_value):
    """ Transform 1d int index to 2d tuple position """
    position = (int(index_value / 3), index_value % 3)
    return position

def position_to_index(index_tuple):
    """ Transform 2d tuple position to 1d int index """
    index = 3 * index_tuple[0] + index_tuple[1]
    return index

