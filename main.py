from data_reader import process_data_and_write_to_csv, read_data_from_csv_to_dict
from data_analyzer import analyze_data
from ml_system import bayes_tuning, svm_tuning, cross_validation_naive_bayes, cross_validation_svm
from experiments import alpha_test_for_bayes, c_test_for_svm, loss_test_for_svm, test_bayes_features, \
    test_svm_features, feature_max_f_test_for_bayes, feature_max_f_test_for_svm, feature_ngram_range_test_for_bayes, \
    feature_ngram_range_test_for_svm, feature_min_df_test_for_bayes, feature_min_df_test_for_svm, \
    feature_max_df_test_for_bayes, feature_max_df_test_for_svm

if __name__ == '__main__':
    # process_data_and_write_to_csv()
    # data = read_data_from_csv_to_dict()
    # analyze_data(data)
    # cross_validation_naive_bayes(True, 0.5, 0.001, 5000, (1, 2), 0.1)
    # cross_validation_svm(True, 0.5, 0.001, 5000, (1, 1), 0.1, 'squared_hinge')
    # bayes_tuning()
    # svm_tuning()

    # #####################Experiments#############################
    # alpha_test_for_bayes()
    # c_test_for_svm()
    # loss_test_for_svm()
    # test_bayes_features()
    test_svm_features()
    # feature_max_f_test_for_bayes()
    # feature_max_f_test_for_svm()
    # feature_ngram_range_test_for_bayes()
    # feature_ngram_range_test_for_svm()
    # feature_min_df_test_for_bayes()
    # feature_min_df_test_for_svm()
    # feature_max_df_test_for_bayes()
    # feature_max_df_test_for_svm()
    pass
