from data_reader import process_data_and_write_to_csv, read_data_from_csv_to_dict
from data_analyzer import analyze_data
from ml_system import naive_bayes, cross_validation_naive_bayes

if __name__ == '__main__':
    # process_data_and_write_to_csv()
    # data = read_data_from_csv_to_dict()
    # analyze_data(data)
    # naive_bayes()
    cross_validation_naive_bayes()
    pass


