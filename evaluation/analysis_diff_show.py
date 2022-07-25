import pandas as pd
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.evaluation.evaluators.cmp_table import DocumentCompareTable
from arekit.common.evaluation.result import BaseEvalResult


def extract_single_diff_table(eval_result, etalon_samples_filepath):
    """ Отображение разницы.
    """
    assert(isinstance(eval_result, BaseEvalResult))

    dataframes = []
    for doc_id, doc_cmp_table in eval_result.iter_dataframe_cmp_tables():
        assert(isinstance(doc_cmp_table, DocumentCompareTable))
        df = doc_cmp_table.DataframeTable
        df.insert(2, "doc_id", [doc_id] * len(df), allow_duplicates=True)
        dataframes.append(df[(df["how_results"] != "NoLabel") &
                             (df["how_orig"].notnull()) &
                             (df["how_results"].notnull()) &
                             (df["comparison"] == False)])

    eval_errors_df = pd.concat(dataframes, axis=0)

    last_column_index = len(eval_errors_df.columns)
    for sample_col in ["text_a", "t_ind", "s_ind", "sent_ind"]:
        eval_errors_df.insert(last_column_index, sample_col, [""] * len(eval_errors_df), allow_duplicates=True)

    # Дополняем содержимым из samples строки с неверно размеченными оценками.
    samples_df = BaseRowsStorage.from_tsv(filepath=etalon_samples_filepath).DataFrame
    for row_id, row in eval_errors_df.iterrows():
        doc_id, source_ind, target_ind = [int(v) for v in row["id_orig"].split("_")]
        sample_rows = samples_df[(samples_df["doc_id"] == doc_id) &
                                 (samples_df["s_ind"] == source_ind) &
                                 (samples_df["t_ind"] == target_ind)]
        sample_row = sample_rows.iloc[0]

        for sample_col in ["s_ind", "t_ind", "text_a", "sent_ind"]:
            eval_errors_df.loc[row_id, sample_col] = sample_row[sample_col]

        # Пост-обработка текста
        text_terms = sample_row["text_a"].lower().split(' ')
        entity_inds = [int(v) for v in sample_row["entities"].split(',')]
        entity_values = sample_row["entity_values"].split(',')
        # заменить на значения сущностей.
        for i, e_ind in enumerate(entity_inds):
            text_terms[e_ind] = entity_values[i].replace(' ', '-')
        # поднять регистр для пары.
        text_terms[source_ind] = text_terms[source_ind].upper()
        text_terms[target_ind] = text_terms[target_ind].upper()
        # усечение по границам для более удобного просмотра области.
        l = max(0, source_ind-10)
        r = min(target_ind+10, len(text_terms)-1)
        eval_errors_df.loc[row_id, "text_a"] = " ".join(text_terms[l:r])

    return eval_errors_df
