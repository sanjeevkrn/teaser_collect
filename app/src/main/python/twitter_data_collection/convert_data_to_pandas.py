import argparse
import codecs
import gzip
import json
import pandas as pd
from pandas.io.json import json_normalize


def setup_argparser():
    my_parser = argparse.ArgumentParser(prog='twitter_read_stream')

    my_parser.add_argument('-v', '--version', action='version', version='Twitter data format transformer')
    my_parser.add_argument("-i",
                           "--input_file",
                           type=str,
                           help="input file containing data. Can be gzipped or plain text",
                           default=None)
    my_parser.add_argument("-o",
                           "--pandas_file",
                           type=str,
                           help="file to write a pickled pandas data frame out to",
                           default=None)

    return my_parser


def load_data(filename):
    """
    Parse the input filename into a list of json dicts containing the data
    :param filename:
    :return:
    """

    if filename.endswith('.gz.'):
        file_handle = codecs.getreader('utf8')(gzip.open(filename, 'rb'))
    else:
        file_handle = codecs.open(filename, 'r', encoding='utf8')

    result = []
    for line_counter, line in enumerate(file_handle):
        try:
            result.append(json.loads(line.strip()))
        except ValueError as v:
            print("unparseable json object at line %s" % line_counter)
            continue
        line_counter += 1
        if line_counter % 10000 == 0:
            print("Processed %s lines..." % line_counter)

    return result


def write_pandas(data, filename):
    """
    writes the given data to a pandas pickled data frame
    :param data:
    :param filename:
    :return:
    """
    try:
        df = json_normalize(data)
        df.to_pickle(filename)
        print("Wrote data to %s" % filename)
    except Exception as err:
        print(err)
        raise


def main():

    parser = setup_argparser()
    args = parser.parse_args()
    print(args)
    data = load_data(args.input_file)

    print(len(data))

    if args.pandas_file:
        write_pandas(data, args.pandas_file)



if __name__ == '__main__':
    main()