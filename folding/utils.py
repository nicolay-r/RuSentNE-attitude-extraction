from collections import OrderedDict


def number_from_string(s):
    assert(isinstance(s, str))

    digit_chars_prefix = []

    for chr in s:
        if chr.isdigit():
            digit_chars_prefix.append(chr)
        else:
            break

    if len(digit_chars_prefix) == 0:
        return None

    return int("".join(digit_chars_prefix))


def create_filenames_by_ids(filenames):
    """ Indexing filenames
    """

    def __create_new_id(default_id):
        new_id = default_id
        while new_id in filenames_by_ids:
            new_id += 1
        return new_id

    default_id = 0

    filenames_by_ids = OrderedDict()
    for fname in filenames:

        doc_id = number_from_string(fname)

        if doc_id is None:
            doc_id = __create_new_id(default_id)
            default_id = doc_id

        filenames_by_ids[doc_id] = fname

    return filenames_by_ids
