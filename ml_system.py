import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


# 1 naive bayes training and test
def naive_bayes_one_time(max_df: float, min_df: float, max_f: float, ngram_range: (int, int), alpha: float):
    # read data from csv
    df = pd.read_csv('cleaned_data.csv', names=['wikipedia_id', 'genre', 'summary'])

    # divide data into train summaries and its labels and text summaries and its labels
    x_train_summaries, x_test_summaries, y_train_genres, y_test_genres = \
        train_test_split(df['summary'], df['genre'], test_size=0.25)

    # convert a collection of text documents to a matrix of token counts
    # max_df - When building the vocabulary ignore terms that have a document frequency strictly
    #   higher than the given threshold
    # max_features - If not None, build a vocabulary that only consider the top max_features ordered by term
    #   frequency across the corpus
    # min_df - When building the vocabulary ignore terms that have a document frequency strictly
    #   lower than the given threshold
    # ngram_range - The lower and upper boundary of the range of n-values for different word
    #   n-grams or char n-grams to be extracted
    count_vectorizer = CountVectorizer(max_df=max_df, max_features=max_f, min_df=min_df, ngram_range=ngram_range)

    # transform a count matrix to a normalized tf or tf-idf representation
    # tf - frequency of word in document
    # tf-idf - frequency of word in document compared to document size
    tfidf_transformer = TfidfTransformer(use_idf=True)
    # alpha - parametr wygładzania krzywej
    naive_bayes_classificator = MultinomialNB(alpha=alpha)

    x_train_counts = count_vectorizer.fit_transform(x_train_summaries)
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

    # creating and teaching the model on given dataset and its labels
    naive_bayes_classificator.fit(x_train_tfidf, y_train_genres)

    # transform summaries for testing
    x_test_transformed = count_vectorizer.transform(x_test_summaries)

    # predicts genres for test summaries
    y_pred = naive_bayes_classificator.predict(x_test_transformed)

    # calculates arithmetic mean of correctness of test summaries predicted genres with actual genres of these summaries
    print("Prediction accuracy mean: " + str(np.mean(y_pred == y_test_genres)))
    print()

    # precision - zdolność modelu do nie przypisywania do streszczenia gatunku jeżeli książka nie jest tego gatunku
    # recall - zdolność modelu do znalezienia wszystkich streszczeń danego gatunku książek
    # f1-score - średnia harmoniczna z precision i recall - 1 to najlepiej, 0 najgorzej
    # support - liczba wystąpień danego gatunku
    print("Report")
    print(classification_report(y_test_genres, y_pred))

    # matrix which shows truthfulness of classification
    # value in cell ij represents amount of summaries which genre was classified as i, but wera classified with j
    # on diagonal should be the greatest value (i == j)
    print("Confusion matrix")
    print(confusion_matrix(y_test_genres, y_pred))


# cross validation naive bayes train and test
def cross_validation_naive_bayes(verbose: bool, max_df: float, min_df: float, max_f: float, ngram_range: (int, int),
                                 alpha: float, use_idf=True) -> (float, float, float):
    df = pd.read_csv('cleaned_data.csv', names=['wikipedia_id', 'genre', 'summary'])

    count_vectorizer = CountVectorizer(max_df=max_df, max_features=max_f, min_df=min_df, ngram_range=ngram_range)
    tfidf_transformer = TfidfTransformer(use_idf=use_idf)
    naive_bayes_classificator = MultinomialNB(alpha=alpha)

    x_train_counts = count_vectorizer.fit_transform(df['summary'])
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

    # cross validates for the model - 10 times
    cv_results = cross_validate(naive_bayes_classificator, x_train_tfidf, df['genre'], cv=10)

    fit_times = float(np.mean(cv_results['fit_time']))
    score_times = float(np.mean(cv_results['score_time']))

    if verbose:
        print('Fit times')
        for elem in cv_results['fit_time']:
            print('\t' + str(round(elem, 5)) + ' s')
        print()

        print('Score times')
        for elem in cv_results['score_time']:
            print('\t' + str(round(elem, 5)) + ' s')
        print()

        print('Test score')
        for elem in cv_results['test_score']:
            print('\t' + str(round(elem * 100, 2)) + ' %')
        print()

        print(np.mean(cv_results['test_score']))

    return float(np.mean(cv_results['test_score'])), fit_times, score_times


# 1 support vector machine training and test
def svm_one_time(max_df: float, min_df: float, max_f: float, ngram_range: (int, int), c: float, loss: str):
    df = pd.read_csv('cleaned_data.csv', names=['wikipedia_id', 'genre', 'summary'])

    x_train_summaries, x_test_summaries, y_train_genres, y_test_genres = \
        train_test_split(df['summary'], df['genre'], test_size=0.25)

    count_vectorizer = CountVectorizer(max_df=max_df, max_features=max_f, min_df=min_df, ngram_range=ngram_range)
    tfidf_transformer = TfidfTransformer(use_idf=True)

    # loss - specifies the loss function (funckja obliczjąca odległość między obecnym i oczekiwamyn wyjściem)
    # C - regularisation parameter (zmniejsza blad przez odpowiednie dopasowanie funkcji i
    #   zapobieganie zjawisku overfitting)
    # dual - False jeżeli wielkośc zbioru > liczby cech
    # class_weight - dostraja wagę klas (etykiet) w zależności od częstości występowania w zbiorze treningowym
    svm_classifier = LinearSVC(C=c, loss=loss, class_weight='balanced')

    x_train_counts = count_vectorizer.fit_transform(x_train_summaries)
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

    svm_classifier.fit(x_train_tfidf, y_train_genres)

    x_test_transformed = count_vectorizer.transform(x_test_summaries)

    y_pred = svm_classifier.predict(x_test_transformed)

    print("Prediction accuracy mean: " + str(np.mean(y_pred == y_test_genres)))
    print()

    print("Report")
    print(classification_report(y_test_genres, y_pred))

    print("Confusion matrix")
    print(confusion_matrix(y_test_genres, y_pred))


# cross validation naive bayes train and test
def cross_validation_svm(verbose: bool, max_df: float, min_df: float, max_f: float, ngram_range: (int, int), c: float,
                         loss: str, use_idf=True) -> (float, float, float):
    df = pd.read_csv('cleaned_data.csv', names=['wikipedia_id', 'genre', 'summary'])

    count_vectorizer = CountVectorizer(max_df=max_df, max_features=max_f, min_df=min_df, ngram_range=ngram_range)
    tfidf_transformer = TfidfTransformer(use_idf=use_idf)
    svm_classificator = LinearSVC(C=c, loss=loss, class_weight='balanced')

    x_train_counts = count_vectorizer.fit_transform(df['summary'])
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

    # cross validates for the model - 10 times
    cv_results = cross_validate(svm_classificator, x_train_tfidf, df['genre'], cv=10)

    fit_times = float(np.mean(cv_results['fit_time']))
    score_times = float(np.mean(cv_results['score_time']))

    if verbose:
        print('Fit times')
        for elem in cv_results['fit_time']:
            print('\t' + str(round(elem, 5)) + ' s')
        print()

        print('Score times')
        for elem in cv_results['score_time']:
            print('\t' + str(round(elem, 5)) + ' s')
        print()

        print('Test score')
        for elem in cv_results['test_score']:
            print('\t' + str(round(elem * 100, 2)) + ' %')
        print()

        print(np.mean(cv_results['test_score']))

    return float(np.mean(cv_results['test_score'])), fit_times, score_times


def bayes_tuning():
    max_df_params = [0.5, 0.7, 0.9]
    min_df_params = [0.001, 0.01, 0.1]
    max_f_params = [1000, 3000, 5000]
    ngram_range_params = [(1, 1), (1, 2), (1, 3)]
    alpha_params = [0.01, 0.1, 1, 2]
    best_mean = 0.0
    best_params = {'max_df': 0.0, 'min_df': 0.0, 'max_f': 0.0, 'ngram_range': (0, 0), 'alpha': 0.0}

    for max_df in max_df_params:
        for min_df in min_df_params:
            for max_f in max_f_params:
                for ngram_range in ngram_range_params:
                    for alpha in alpha_params:
                        new_mean = cross_validation_naive_bayes(False, max_df, min_df, max_f, ngram_range, alpha)
                        print(max_df, min_df, max_f, ngram_range, alpha)
                        if new_mean > best_mean:
                            best_mean = new_mean
                            best_params = {'max_df': max_df, 'min_df': min_df,
                                           'max_f': max_f, 'ngram_range': ngram_range,
                                           'alpha': alpha}

    print('-------------------------------------- Tuning multinomial naive bayes')
    print("Best mean: " + str(round(best_mean, 3)))
    print("Parameters for best mean: ")
    for key in best_params:
        print('\t' + key + ': ' + str(best_params[key]))


def svm_tuning():
    max_df_params = [0.5, 0.7, 0.9]
    min_df_params = [0.001, 0.01, 0.1]
    max_f_params = [1000, 3000, 5000]
    ngram_range_params = [(1, 1), (1, 2), (1, 3)]
    c_params = [0.1, 1, 2]
    loss_params = ['hinge', 'squared_hinge']
    best_mean = 0.0
    best_params = {'max_df': 0.0, 'min_df': 0.0, 'max_f': 0, 'ngram_range': (0, 0), 'c': 0, 'loss': 'hinge'}
    for max_df in max_df_params:
        for min_df in min_df_params:
            for max_f in max_f_params:
                for ngram_range in ngram_range_params:
                    for c in c_params:
                        for loss in loss_params:
                            new_mean = cross_validation_svm(False, max_df, min_df, max_f, ngram_range, c, loss)
                            print(max_df, min_df, max_f, ngram_range, alpha)
                            if new_mean > best_mean:
                                best_mean = new_mean
                                best_params['max_df'] = max_df
                                best_params['min_df'] = min_df
                                best_params['max_f'] = max_f
                                best_params['ngram_range'] = ngram_range
                                best_params['c'] = c
                                best_params['loss'] = loss

    print('-------------------------------------- Tuning svm')
    print("Best mean: " + str(round(best_mean, 3)))
    print("Parameters for best mean: ")
    for key in best_params:
        print("\t" + key + ": " + str(best_params[key]))
    print()
