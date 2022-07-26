import pandas as pd
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.evaluation.evaluators.cmp_table import DocumentCompareTable
from arekit.common.evaluation.result import BaseEvalResult


def __post_text_processing(sample_row, source_ind, target_ind, window_crop=10):
    """ Пост-обработка текста для лучшего понимания возможных возникаюцих проблем в
        анализе текста.
    """
    assert("text_a" in sample_row)
    assert("entities" in sample_row)
    assert("entity_values" in sample_row)
    assert("entity_types" in sample_row)

    text_terms = sample_row["text_a"].lower().split(' ')

    entity_inds = [int(v) for v in sample_row["entities"].split(',')]
    entity_values = sample_row["entity_values"].split(',')
    entity_types = sample_row["entity_types"].split(',')

    # заменить на значения сущностей.
    for i, e_ind in enumerate(entity_inds):
        text_terms[e_ind] = entity_values[i].replace(' ', '-')

    # поднять регистр для пары.
    text_terms[source_ind] = text_terms[source_ind].upper() + "-[{}]".format(
        entity_types[entity_inds.index(source_ind)].replace(' ', '-'))
    text_terms[target_ind] = text_terms[target_ind].upper() + "-[{}]".format(
        entity_types[entity_inds.index(target_ind)].replace(' ', '-'))

    # усечение по границам для более удобного просмотра области.
    left_participant = min(source_ind, target_ind)
    right_participant = max(source_ind, target_ind)
    crop_left = max(0, left_participant - window_crop)
    crop_right = min(right_participant + window_crop, len(text_terms))

    return " ".join(text_terms[crop_left:crop_right])


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
    eval_errors_df.reset_index(inplace=True)

    columns_to_copy = ["entity_types", "text_a", "t_ind", "s_ind", "sent_ind"]
    last_column_index = len(eval_errors_df.columns)
    for sample_col in columns_to_copy:
        eval_errors_df.insert(last_column_index, sample_col, [""] * len(eval_errors_df), allow_duplicates=True)

    # Дополняем содержимым из samples строки с неверно размеченными оценками.
    samples_df = BaseRowsStorage.from_tsv(filepath=etalon_samples_filepath).DataFrame
    for row_id, eval_row in eval_errors_df.iterrows():

        doc_id, ctx_id, source_ind, target_ind = [int(v) for v in eval_row["id_orig"].split("_")]
        sample_rows = samples_df[(samples_df["doc_id"] == doc_id) &
                                 (samples_df["sent_ind"] == ctx_id) &
                                 (samples_df["s_ind"] == source_ind) &
                                 (samples_df["t_ind"] == target_ind)]

        sample_row = sample_rows.iloc[0]

        for sample_col in columns_to_copy:
            eval_errors_df.at[row_id, sample_col] = sample_row[sample_col]

        eval_errors_df.at[row_id, "text_a"] = __post_text_processing(
            sample_row=sample_row, source_ind=source_ind, target_ind=target_ind)

    return eval_errors_df
