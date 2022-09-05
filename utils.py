from evaluation.calc_labels import calculate_samples_count_per_label


def show_stat_for_samples(samples_filepath, no_label_uint):
    print("Evaluate for {dataset}".format(dataset=samples_filepath))
    labels_stat = calculate_samples_count_per_label(samples_filepath, no_label_uint=no_label_uint)
    print(labels_stat)
