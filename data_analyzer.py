# analyzes data
def analyze_data(data: dict[int, dict[str, str]]):
    genres_count = count_genres(data)
    sorted_counts = sorted(genres_count.items(), key=lambda x: x[1], reverse=True)
    print("Genre list with counts: ")
    for item in sorted_counts:
        print("\t" + item[0] + " " + str(item[1]))

    print('Avg amount of letters in summary: ' + str(calculate_avg_summary_length_in_letters(data)))

    print('Avg amount of words in summary: ' + str(calculate_avg_summary_length_in_words(data)))

    calculated_words_occurrences = calculate_words(data)
    sorted_counts = sorted(calculated_words_occurrences.items(), key=lambda x: x[1], reverse=True)
    print("Top 15 words with most occurrences: ")
    for i in range(0, 15):
        print("\t" + sorted_counts[i][0] + " " + str(sorted_counts[i][1]))

    sum_in_words = calculate_summaries_length_in_words(data)
    sorted_counts = sorted(sum_in_words.items(), key=lambda x: x[1], reverse=True)
    print('The most common summary length in words: ' + str(sorted_counts[0][0]))

    sorted_counts = sorted(sum_in_words.items(), key=lambda x: x[0], reverse=True)
    print('The highest summary length in words: ' + str(sorted_counts[0][0]))
    print('The lowest summary length in words: ' + str(sorted_counts[len(sorted_counts) - 1][0]))

    sum_in_letters = calculate_the_summaries_length_in_letters(data)
    sorted_counts = sorted(sum_in_letters.items(), key=lambda x: x[1], reverse=True)
    print('The most common summary length in letters: ' + str(sorted_counts[0][0]))

    sorted_counts = sorted(sum_in_letters.items(), key=lambda x: x[0], reverse=True)
    print('The highest summary length in letters: ' + str(sorted_counts[0][0]))
    print('The lowest summary length in letters: ' + str(sorted_counts[len(sorted_counts) - 1][0]))


# counts genres occurrences among books
def count_genres(temp_dict: dict[int, dict[str, str]]) -> dict[str, int]:
    genre_count = {}

    for key in temp_dict:
        if temp_dict[key]['genre'] in genre_count:
            genre_count[temp_dict[key]['genre']] += 1
        else:
            genre_count[temp_dict[key]['genre']] = 1

    return genre_count


# calculates avg length of summary in letters
def calculate_avg_summary_length_in_letters(temp_dict: dict[int, dict[str, str]]) -> [int]:
    sum_of_letters = 0

    for key in temp_dict.keys():
        sum_of_letters += len(temp_dict[key]['summary'])

    avg = sum_of_letters / len(temp_dict)
    return int(round(avg, 0))


# calculates avg length of summary in words
def calculate_avg_summary_length_in_words(temp_dict: dict[int, dict[str, str]]) -> [int]:
    sum_of_words = 0

    for key in temp_dict.keys():
        words = temp_dict[key]['summary'].split()
        for word in words:
            if if_is_word(word):
                sum_of_words += 1

    avg = sum_of_words / len(temp_dict)

    return int(round(avg, 0))


# calculate amount of words occurrences
def calculate_words(temp_dict: dict[int, dict[str, str]]) -> dict[str, int]:
    counts_of_words = {}

    for book in temp_dict.values():
        words = book['summary'].split()
        for word in words:
            if word.lower() in counts_of_words:
                counts_of_words[word.lower()] += 1
            elif if_is_word(word.lower()):
                counts_of_words[word.lower()] = 1

    return counts_of_words


# helper method to determine if text is a word
def if_is_word(text: str) -> bool:
    for i in range(len(text)):
        if (text[i].lower() < "a" or text[i].lower() > "z") and text[i] != "'":
            return False
    return True


# calculate the most common length of summary in words that occurred
def calculate_summaries_length_in_words(temp_dict: dict[int, dict[str, str]]) -> dict[int, int]:
    counts_of_words = {}

    for book in temp_dict.values():
        words = book['summary'].split()
        words_counter = 0
        for word in words:
            if if_is_word(word):
                words_counter += 1

        if words_counter in counts_of_words:
            counts_of_words[words_counter] += 1
        else:
            counts_of_words[words_counter] = 1

    return counts_of_words


# calculate the most common length in letters of summary that occurred
def calculate_the_summaries_length_in_letters(temp_dict: dict[int, dict[str, str]]) -> dict[int, int]:
    counts_of_words = {}

    for book in temp_dict.values():
        count = len(book['summary'])
        if count in counts_of_words:
            counts_of_words[count] += 1
        else:
            counts_of_words[count] = 1

    return counts_of_words

