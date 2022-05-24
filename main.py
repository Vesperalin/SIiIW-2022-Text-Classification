from data_reader import process_data_and_write_to_csv, read_data_from_csv
from data_analyzer import analyze_data

if __name__ == '__main__':
    # process_data_and_write_to_csv()
    data = read_data_from_csv()
    analyze_data(data)
