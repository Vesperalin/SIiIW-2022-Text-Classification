import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.svm import LinearSVC


from ml_system import cross_validation_naive_bayes, cross_validation_svm


def alpha_test_for_bayes():
    alpha_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    for alpha in alpha_values:
        print("-------------------------------------------------------------------------- " + str(alpha))
        result = cross_validation_naive_bayes(False, 0.5, 0.001, 5000, (1, 1), alpha)
        print('Test score: ' + str(round(result[0] * 100, 1)) + "%")
        print('Fit times: ' + str(round(result[1], 3)))
        print('Score times: ' + str(round(result[2], 3)))


def c_test_for_svm():
    c_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    for c in c_values:
        print("-------------------------------------------------------------------------- " + str(c))
        result = cross_validation_svm(False, 0.5, 0.001, 5000, (1, 1), c, 'squared_hinge')
        print('Test score: ' + str(round(result[0] * 100, 1)) + "%")
        print('Fit times: ' + str(round(result[1], 3)))
        print('Score times: ' + str(round(result[2], 3)))


def loss_test_for_svm():
    loss_values = ['hinge', 'squared_hinge']
    for loss in loss_values:
        print("-------------------------------------------------------------------------- " + loss)
        result = cross_validation_svm(False, 0.5, 0.001, 5000, (1, 1), 0.1, loss)
        print('Test score: ' + str(round(result[0] * 100, 1)) + "%")
        print('Fit times: ' + str(round(result[1], 3)))
        print('Score times: ' + str(round(result[2], 3)))


def test_bayes_features():
    feature_max_df_test_for_bayes()
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    feature_min_df_test_for_bayes()
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    feature_max_f_test_for_bayes()
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    feature_ngram_range_test_for_bayes()


def feature_max_df_test_for_bayes():
    max_df_params = [0.1, 0.5, 0.9]
    for max_df in max_df_params:
        print("-------------------------------------------------------------------------- " + str(max_df))
        result = cross_validation_naive_bayes(False, max_df, 0.001, 5000, (1, 2), 0.1)
        print('Test score: ' + str(round(result[0] * 100, 1)) + "%")
        print('Fit times: ' + str(round(result[1], 3)))
        print('Score times: ' + str(round(result[2], 3)))


def feature_min_df_test_for_bayes():
    min_df_params = [0.001, 0.01, 0.1]
    for min_df in min_df_params:
        print("-------------------------------------------------------------------------- " + str(min_df))
        result = cross_validation_naive_bayes(False, 0.5, min_df, 5000, (1, 2), 0.1)
        print('Test score: ' + str(round(result[0] * 100, 1)) + "%")
        print('Fit times: ' + str(round(result[1], 3)))
        print('Score times: ' + str(round(result[2], 3)))


def feature_max_f_test_for_bayes():
    max_f_params = [1000, 3000, 5000]
    for max_f in max_f_params:
        print("-------------------------------------------------------------------------- " + str(max_f))
        result = cross_validation_naive_bayes(False, 0.5, 0.001, max_f, (1, 2), 0.1)
        print('Test score: ' + str(round(result[0] * 100, 1)) + "%")
        print('Fit times: ' + str(round(result[1], 3)))
        print('Score times: ' + str(round(result[2], 3)))


def feature_ngram_range_test_for_bayes():
    ngram_range_params = [(1, 1), (1, 2), (1, 3)]
    for ngram_range in ngram_range_params:
        print("-------------------------------------------------------------------------- " + str(ngram_range))
        result = cross_validation_naive_bayes(False, 0.5, 0.001, 5000, ngram_range, 0.1)
        print('Test score: ' + str(round(result[0] * 100, 1)) + "%")
        print('Fit times: ' + str(round(result[1], 3)))
        print('Score times: ' + str(round(result[2], 3)))


def test_svm_features():
    feature_max_df_test_for_svm()
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    feature_min_df_test_for_svm()
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    feature_max_f_test_for_svm()
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    feature_ngram_range_test_for_svm()


def feature_max_df_test_for_svm():
    max_df_params = [0.1, 0.5, 0.9]
    for max_df in max_df_params:
        print("-------------------------------------------------------------------------- " + str(max_df))
        result = cross_validation_svm(False, max_df, 0.001, 5000, (1, 1), 0.1, 'squared_hinge')
        print('Test score: ' + str(round(result[0] * 100, 1)) + "%")
        print('Fit times: ' + str(round(result[1], 3)))
        print('Score times: ' + str(round(result[2], 3)))


def feature_min_df_test_for_svm():
    min_df_params = [0.001, 0.01, 0.1]
    for min_df in min_df_params:
        print("-------------------------------------------------------------------------- " + str(min_df))
        result = cross_validation_svm(False, 0.5, min_df, 5000, (1, 1), 0.1, 'squared_hinge')
        print('Test score: ' + str(round(result[0] * 100, 1)) + "%")
        print('Fit times: ' + str(round(result[1], 3)))
        print('Score times: ' + str(round(result[2], 3)))


def feature_max_f_test_for_svm():
    max_f_params = [1000, 3000, 5000]
    for max_f in max_f_params:
        print("-------------------------------------------------------------------------- " + str(max_f))
        result = cross_validation_svm(False, 0.5, 0.001, max_f, (1, 1), 0.1, 'squared_hinge')
        print('Test score: ' + str(round(result[0] * 100, 1)) + "%")
        print('Fit times: ' + str(round(result[1], 3)))
        print('Score times: ' + str(round(result[2], 3)))


def feature_ngram_range_test_for_svm():
    ngram_range_params = [(1, 1), (1, 2), (1, 3)]
    for ngram_range in ngram_range_params:
        print("-------------------------------------------------------------------------- " + str(ngram_range))
        result = cross_validation_svm(False, 0.5, 0.001, 5000, ngram_range, 0.1, 'squared_hinge')
        print('Test score: ' + str(round(result[0] * 100, 1)) + "%")
        print('Fit times: ' + str(round(result[1], 3)))
        print('Score times: ' + str(round(result[2], 3)))


def naive_bayes_cross_val(df, max_df, min_df, max_f, ngram_range, alpha):
    count_vectorizer = CountVectorizer(max_df=max_df, max_features=max_f, min_df=min_df, ngram_range=ngram_range)
    tfidf_transformer = TfidfTransformer(use_idf=True)
    naive_bayes_classificator = MultinomialNB(alpha=alpha)

    x_train_counts = count_vectorizer.fit_transform(df['summary'])
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

    cv_results = cross_validate(naive_bayes_classificator, x_train_tfidf, df['genre'], cv=10)

    return float(np.mean(cv_results['test_score']))


def test_parameters_for_bayes_for_different_size_of_data_set():
    df = pd.read_csv('cleaned_data.csv', names=['wikipedia_id', 'genre', 'summary'])

    alpha_params = [0.001, 0.01, 0.1, 1, 10, 100]
    sizes = [200, 500, 1000, 2000, 3000, 4000]

    for size in sizes:
        best_mean = 0.0
        best_alpha = 0

        for alpha in alpha_params:
            new_mean = naive_bayes_cross_val(df.iloc[1:size], 0.5, 0.001, 5000, (1, 2), alpha)
            print(alpha, size)

            if new_mean > best_mean:
                best_mean = new_mean
                best_alpha = alpha

        print("###########################################################################")
        print("Size: " + str(size) + ", Alpha: " + str(best_alpha) + ", Best mean: " + str(round(best_mean, 3) * 100) + "%")


def linear_svc_cross_val(df, max_df, min_df, max_f, ngram_range, c, loss):
    count_vectorizer = CountVectorizer(max_df=max_df, max_features=max_f, min_df=min_df, ngram_range=ngram_range)
    tfidf_transformer = TfidfTransformer(use_idf=True)
    svm_classificator = LinearSVC(C=c, loss=loss, class_weight='balanced')

    x_train_counts = count_vectorizer.fit_transform(df['summary'])
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

    cv_results = cross_validate(svm_classificator, x_train_tfidf, df['genre'], cv=10)

    return float(np.mean(cv_results['test_score']))


def test_parameters_for_linear_svc_for_different_size_of_data_set():
    df = pd.read_csv('cleaned_data.csv', names=['wikipedia_id', 'genre', 'summary'])

    c_params = [0.001, 0.01, 0.1, 1, 10, 100]
    loss_params = ['hinge', 'squared_hinge']
    sizes = [200, 500, 1000, 2000, 3000, 4000]

    for size in sizes:
        best_mean = 0.0
        best_c = 0
        best_loss = ''

        for c in c_params:
            for loss in loss_params:
                new_mean = linear_svc_cross_val(df.iloc[1:size], 0.5, 0.001, 5000, (1, 2), c, loss)
                print(c, loss, size)

                if new_mean > best_mean:
                    best_mean = new_mean
                    best_c = c
                    best_loss = loss

        print("###########################################################################")
        print("Size: " + str(size) + ", C: " + str(best_c) + ", Loss: " + best_loss + ", Best mean: " + str(round(best_mean, 3) * 100) + "%")
