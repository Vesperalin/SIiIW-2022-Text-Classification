import json
from copy import deepcopy
import pandas as pd
import csv


# reads data from file, transforms and saves to csv file - merges all helper methods into one
def process_data_and_write_to_csv():
    raw_data = read_data()
    data = []

    print('Initial amount of lines: ' + str(len(raw_data)))

    for index in range(len(raw_data)):
        try:
            data.append(utf_16_encoding_to_8_fix(raw_data[index]))
        except ValueError:
            pass

    print('Amount of lines after deleting lines with bad format: ' + str(len(data)))

    data = transform_data_to_dict(data)
    data = remove_to_wide_or_to_specific_genres(data)
    data = remove_books_with_too_many_genres(data)
    data = transform_temporary_dictionary_to_target_dictionary(data)
    save_data_to_csv(data)

    print('Amount of lines after cleaning and choosing genres: ' + str(len(data)))


# reads data from file and returns as list of lines - strings
def read_data() -> list[str]:
    with open('booksummaries/booksummaries.txt', 'r', encoding="utf-8") as file:
        return file.readlines()


# https://reddit.fun/1604/how-print-strings-with-unicode-escape-characters-correctly
# https://realpython.com/python-enumerate/#using-pythons-enumerate
# encodes special characters
# may throw an exception when \ used in text - so we catch the exception and do not use the line
def utf_16_encoding_to_8_fix(data: str) -> str:
    data = list(data)
    for index, value in enumerate(data):
        if value == '\\':
            val = ''.join([data[index + k + 2] for k in range(4)])
            for k in range(5):
                data.pop(index)
            data[index] = str(chr(int(val, 16)))
    return ''.join(data)


# transforms data to dict where: key - Wiki ID, value - dict with: title, genres(list) and summary
def transform_data_to_dict(data: list[str]) -> dict[int, dict[str, list[str]]]:
    temporary_data = {}
    for line in data:
        line = line.split("\t")

        # deletes leading and trailing whitespaces from elements that will be used
        # line[0] - ID from Wiki, line[2] - title, line[5] - genres, line[6] - summary
        line[0] = line[0].strip()
        line[2] = line[2].strip()
        line[5] = line[5].strip()
        line[6] = line[6].strip()

        # delete elements that doesnt have Wiki ID, title, genres, summary or have bad content
        if line[0] != '' and line[2] != '' and line[5] != '' and line[6] != '' and \
                line[6].count('To be added.') == 0 and line[6].count('=') == 0 and line[6].count("\\") == 0 and \
                line[6].count('/') == 0 and line[6].count('<!') == 0 and line[6].count('<') == 0 and \
                line[6].count('&nbsp') == 0 and line[6].count('&mdash') == 0 and line[6].count('&ndash') == 0 and \
                line[6].count('&#') == 0 and line[6].count('http') == 0 and \
                line[6].count('Plot outline description') == 0:

            nested_dict = {'title': [line[2]],
                           'summary': [line[6]],
                           'genres': [genre for genre in json.loads(line[5]).values()]}

            temporary_data[int(line[0])] = nested_dict

    return temporary_data


# counts genres occurrences among books
def count_genres(temp_dict: dict[int, dict[str, list[str]]]) -> dict[str, int]:
    genre_count = {}
    for element in temp_dict.values():
        for genre in element['genres']:
            if genre in genre_count:
                genre_count[genre] += 1
            else:
                genre_count[genre] = 1

    return genre_count


# deletes genres with too wide or too specific genres from book genres
# also deletes books with no genres left
def remove_to_wide_or_to_specific_genres(temp_dict: dict[int, dict[str, list[str]]]) -> dict[int, dict[str, list[str]]]:
    genres_count = count_genres(temp_dict)

    for book in temp_dict.values():
        for genre in deepcopy(book['genres']):
            if genres_count[genre] < 1000 or genres_count[genre] > 3000:
                book['genres'].remove(genre)

    tmp = deepcopy(temp_dict)
    for key in tmp:
        if len(temp_dict[key]['genres']) == 0:
            temp_dict.pop(key)

    return temp_dict


# remove books that have more than one genre from the dictionary
def remove_books_with_too_many_genres(temp_dict: dict[int, dict[str, list[str]]]) -> dict[int, dict[str, list[str]]]:
    genres_count = count_genres(temp_dict)
    tmp = deepcopy(temp_dict)

    for key in tmp:
        genres_of_book = set(temp_dict[key]['genres'])
        intersection = genres_of_book.intersection(genres_count.keys())

        if len(list(intersection)) > 1:
            temp_dict.pop(key)

    counts = count_genres(temp_dict)
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    print("Genre list with counts: ")
    for item in sorted_counts:
        print("\t" + item[0] + " " + str(item[1]))

    return temp_dict


# transforms temporary dictionary to target dictionary
def transform_temporary_dictionary_to_target_dictionary(temp_dict: dict[int, dict[str, list[str]]]) -> dict[int, dict[str, str]]:
    final_dict = {}

    for key in temp_dict.keys():
        final_dict[key] = {'title': temp_dict[key]['title'][0],
                           'genre': temp_dict[key]['genres'][0],
                           'summary': temp_dict[key]['summary'][0]}

    return final_dict


# saves cleaned data to csv file
def save_data_to_csv(data: dict[int, dict[str, str]]):
    data_to_save = []

    for key in data:
        data_to_save.append([str(key), data[key]['title'], data[key]['genre'], data[key]['summary']])

    with open("cleaned_data.csv", "w", encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data_to_save)


# reads cleaned data from csv and saves to dictionary
def read_data_from_csv_to_dict() -> dict[int, dict[str, str]]:
    df = pd.read_csv('cleaned_data.csv', header=None)
    data = {}

    # indexes: 0 - Wiki ID, 1, title, 2 - genre, 3 - summary
    for line_index in df.index:
        data[int(df.loc[line_index, 0])] = {
            'title': str(df.loc[line_index, 1]),
            'genre': str(df.loc[line_index, 2]),
            'summary': str(df.loc[line_index, 3])
        }

    return data
