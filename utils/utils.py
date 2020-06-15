import argparse
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',
                        metavar='config_file',
                        help='Configuration file that defines action to perform')

    return parser.parse_args()


def load_config_file(config_file):
    try:
        with open(config_file, 'r') as file:
            yaml_file = yaml.load(file, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print("Configuration {} file not found".format(config_file))
        exit()

    return yaml_file
