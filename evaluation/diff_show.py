import pandas as pd
from arekit.common.evaluation.evaluators.cmp_table import DocumentCompareTable
from arekit.common.evaluation.result import BaseEvalResult


def extract_single_diff_table(result):
    """ Отображение разницы.
    """
    assert(isinstance(result, BaseEvalResult))

    dataframes = []
    for doc_id, doc_cmp_table in result.iter_dataframe_cmp_tables():
        assert(isinstance(doc_cmp_table, DocumentCompareTable))
        df = doc_cmp_table.DataframeTable
        df.insert(2, "doc_id", [doc_id] * len(df), allow_duplicates=True)
        dataframes.append(df[(df["how_results"] != "NoLabel") &
                             (df["how_orig"].notnull()) &
                             (df["how_results"].notnull()) &
                             (df["comparison"] == False)])

    return pd.concat(dataframes, axis=0)
