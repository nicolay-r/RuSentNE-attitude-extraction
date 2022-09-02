from SentiNEREL.folding.fixed import create_fixed_folding


class FoldingFactory:
    """ Factory of the variety types of the splits that
        are considered within the present experiments.
    """

    @staticmethod
    def create_fixed_folding(fixed_split_filepath, limit=None):
        """ Файл к фиксированному разбиению.
            Можно ограничить число документов, чтобы например потестировать. (limit)
        """

        train_filenames, test_filenames = FoldingFactory.__read_train_test(fixed_split_filepath)
        if limit is not None:
            train_filenames = train_filenames[:limit]
            test_filenames = test_filenames[:limit]
        filenames_by_ids, data_folding = create_fixed_folding(train_filenames=train_filenames,
                                                              test_filenames=test_filenames)

        return filenames_by_ids, data_folding

    @staticmethod
    def __read_train_test(filepath):
        with open(filepath, "r") as f:
            parts = []
            for line in f.readlines():
                parts.append(line.strip().split(','))
        return parts[0], parts[1]
