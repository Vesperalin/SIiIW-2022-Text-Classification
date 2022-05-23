import json
from copy import deepcopy


# reads data from file, transforms and saves to csv file
# 9 records are skipped, because they contain unwanted signs
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

    data = data_transformer(data)
    genres_count = count_genres(data)
    data = filter_genres(data, genres_count)

    genres_count = count_genres(data)
    sorted_counts = sorted(genres_count.items(), key=lambda x: x[1], reverse=True)
    for item in sorted_counts:
        print(item[0] + " " + str(item[1]))

    print('Amount of lines after deleting lines with no genres: ' + str(len(data)))


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


# transforms data to dict: key - Wiki ID, value - dict with: title, genres and summary
def data_transformer(data: list[str]) -> dict[int, dict[str, list[str]]]:
    temporary_data = {}
    for line in data:
        line = line.split("\t")

        # deletes leading and trailing whitespaces from elements that will be used
        line[2] = line[2].strip()
        line[5] = line[5].strip()
        line[6] = line[6].strip()

        # delete elements that doesnt have Wiki ID, title, genres or summary
        if line[0] != '' and line[2] != '' and line[5] != '' and line[6] != '' and line[6] != 'To be added.' and\
                line[6].count('=') == 0 and line[6].count("\\") == 0 and line[6].count('/') == 0 and \
                line[6].count('<!') == 0 and line[6].count('<') == 0 and line[6].count('&nbsp') == 0 and \
                line[6].count('&mdash') == 0 and line[6].count('&ndash') == 0 and line[6].count('&#') == 0 and \
                line[6].count('http') == 0:
            # line[0] - ID from Wiki, line[2] - title, line[5] - genres, line[6] - summary
            # temporary_data[int(line[0])] = [line[index] for index in (2, 5, 6)]
            nested_dict = {'title': line[2], 'summary': line[6],
                           'genres': [genre for genre in json.loads(line[5]).values()]}
            temporary_data[int(line[0])] = nested_dict

    print('Amount of lines after deleting lines with wrong data: ' + str(len(temporary_data)))

    return temporary_data


def count_genres(temp_dict: dict[int, dict[str, list[str]]]) -> dict[str, int]:
    genre_count = {}
    for element in temp_dict.values():
        for genre in element['genres']:
            if genre in genre_count:
                genre_count[genre] += 1
            else:
                genre_count[genre] = 1

    return genre_count


# deletes genres with too wide or too specific genres
def filter_genres(temp_dict: dict[int, dict[str, list[str]]], genre_count: dict[str, int]) -> dict[int, dict[str, list[str]]]:
    for book in temp_dict.values():
        for genre in deepcopy(book['genres']):
            if genre_count[genre] < 500 or genre_count[genre] > 3000:
                book['genres'].remove(genre)

    tmp = deepcopy(temp_dict)
    for key in tmp:
        if len(temp_dict[key]['genres']) == 0:
            temp_dict.pop(key)

    return temp_dict

