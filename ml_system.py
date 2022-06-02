import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB


# 1 naive bayes training and test
def naive_bayes():
    # read data from csv
    df = pd.read_csv('cleaned_data.csv', names=['wikipedia_id', 'genre', 'summary'])

    # divide data into train summaries and its labels and text summaries and its labels
    x_train_summaries, x_test_summaries, y_train_genres, y_test_genres = \
        train_test_split(df['summary'], df['genre'], test_size=0.1)

    count_vectorizer = CountVectorizer(max_df=0.5, max_features=5000, min_df=0.01, ngram_range=(1, 2))
    tfidf_transformer = TfidfTransformer(use_idf=True)
    naive_bayes_classificator = MultinomialNB(alpha=0.1)

    # convert a collection of text documents to a matrix of token counts
    x_train_counts = count_vectorizer.fit_transform(x_train_summaries)

    # transform a count matrix to a normalized tf or tf-idf representation
    # tf - frequency of word in document
    # tf-idf - frequency of word in document compared to document size
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

    # creating and teaching the model on given dataset and its labels
    naive_bayes_classificator.fit(x_train_tfidf, y_train_genres)

    x_test_transformed = count_vectorizer.transform(x_test_summaries)

    # predicts genres for test summaries
    y_pred = naive_bayes_classificator.predict(x_test_transformed)

    # calculates arithmetic mean of correctness of test summaries predicted genres with actual genres of these summaries
    print(np.mean(y_pred == y_test_genres))

    # TODO zmienić na angielski
    # precision - zdolność modelu do nie przypisywania do streszczenia gatunku jeżeli książka nie jest tego gatunku
    # recall - zdolność modelu do znalezienia wszystkich streszczeń danego gatunku książek
    # f1-score - średnia harmoniczna z precision i recall - 1 to najlepiej, 0 najgorzej
    # support - liczba wystąpień danego gatunku
    print(classification_report(y_test_genres, y_pred))

    # matrix which shows truthfulness of classification
    # value in cell ij represents amount of summaries which genre was classified as i, but wera classified with j
    # on diagonal should be the greatest value (i == j)
    print(confusion_matrix(y_test_genres, y_pred))


# cross validation naive bayes train and test
def cross_validation_naive_bayes():
    # read data from csv
    df = pd.read_csv('cleaned_data.csv', names=['wikipedia_id', 'genre', 'summary'])

    count_vectorizer = CountVectorizer(max_df=0.5, max_features=5000, min_df=0.01, ngram_range=(1, 2))
    tfidf_transformer = TfidfTransformer(use_idf=True)
    naive_bayes_classificator = MultinomialNB(alpha=0.1)

    # convert a collection of text documents to a matrix of token counts
    x_train_counts = count_vectorizer.fit_transform(df['summary'])

    # transform a count matrix to a normalized tf or tf-idf representation
    # tf - frequency of word in document
    # tf-idf - frequency of word in document compared to document size
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

    # cross validates for the model - 10 times
    cv_results = cross_validate(naive_bayes_classificator, x_train_tfidf, df['genre'], cv=10)

    print('Fit times')
    for elem in cv_results['fit_time']:
        print('\t' + str(round(elem, 5)) + ' s')

    print('Score times')
    for elem in cv_results['score_time']:
        print('\t' + str(round(elem, 5)) + ' s')

    print('Test score')
    for elem in cv_results['test_score']:
        print('\t' + str(round(elem * 100, 2)) + ' %')

    print(np.mean(cv_results['test_score']))
