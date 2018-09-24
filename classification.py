from data_preparaton import *
from models import bisect_model
from time import monotonic
import sys
import argparse
import logging

log = logging.getLogger(__name__)
print("This is the name of the script: ", sys.argv[0])
print("Number of arguments: ", len(sys.argv))
print("The arguments are: ", str(sys.argv))
print('version is', sys.version)

parser = argparse.ArgumentParser()
parser.add_argument("x", type=str, help="the input file")

args = parser.parse_args()
input_path = args.x
start_time = monotonic()


def main():
    data = data_preparation(input_path)
    log.info('Data has been pre_processed')
    bi_predictions = bisect_model(data)
    log.info('BisectingKMeans has been trained')
    #TODO create prediction models


end_time = monotonic()
print(end_time - start_time)


if __name__ == "__main__":
    main()