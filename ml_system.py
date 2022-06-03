import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


# invoke all naive bayes methods
def naive_bayes():
    pass


# 1 naive bayes training and test
def naive_bayes_one_time():
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
    count_vectorizer = CountVectorizer(max_df=0.5, max_features=5000, min_df=0.01, ngram_range=(1, 2))

    # transform a count matrix to a normalized tf or tf-idf representation
    # tf - frequency of word in document
    # tf-idf - frequency of word in document compared to document size
    tfidf_transformer = TfidfTransformer(use_idf=True)
    # alpha - parametr wygładzania krzywej
    naive_bayes_classificator = MultinomialNB(alpha=0.1)

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
def cross_validation_naive_bayes():
    df = pd.read_csv('cleaned_data.csv', names=['wikipedia_id', 'genre', 'summary'])

    count_vectorizer = CountVectorizer(max_df=0.5, max_features=5000, min_df=0.01, ngram_range=(1, 2))
    tfidf_transformer = TfidfTransformer(use_idf=True)
    naive_bayes_classificator = MultinomialNB(alpha=0.1)

    x_train_counts = count_vectorizer.fit_transform(df['summary'])
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

    # cross validates for the model - 10 times
    cv_results = cross_validate(naive_bayes_classificator, x_train_tfidf, df['genre'], cv=10)

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


# 1 support vector machine training and test
def svm_one_time():
    df = pd.read_csv('cleaned_data.csv', names=['wikipedia_id', 'genre', 'summary'])

    x_train_summaries, x_test_summaries, y_train_genres, y_test_genres = \
        train_test_split(df['summary'], df['genre'], test_size=0.25)

    count_vectorizer = CountVectorizer(max_df=0.5, max_features=5000, min_df=0.01, ngram_range=(1, 2))
    tfidf_transformer = TfidfTransformer(use_idf=True)
    # loss - specifies the loss function (funckja obliczjąca odległość między obecnym i oczekiwamyn wyjściem)
    # C - regularisation parameter (zmniejsza blad przez odpowiednie dopasowanie funkcji i
    #   zapobieganie zjawisku overfitting)
    # dual - False jeżeli wielkośc zbioru > liczby cech
    # class_weight - dostraja wagę klas (etykiet) w zależności od częstości występowania w zbiorze treningowym
    svm_classifier = LinearSVC(C=1, loss='squared_hinge', class_weight='balanced', dual=False)

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
def cross_validation_svm():
    df = pd.read_csv('cleaned_data.csv', names=['wikipedia_id', 'genre', 'summary'])

    count_vectorizer = CountVectorizer(max_df=0.5, max_features=5000, min_df=0.01, ngram_range=(1, 2))
    tfidf_transformer = TfidfTransformer(use_idf=True)
    svm_classificator = LinearSVC(C=1, loss='squared_hinge', class_weight='balanced', dual=False)

    x_train_counts = count_vectorizer.fit_transform(df['summary'])
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

    # cross validates for the model - 10 times
    cv_results = cross_validate(svm_classificator, x_train_tfidf, df['genre'], cv=10)

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

