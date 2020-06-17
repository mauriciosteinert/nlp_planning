import argparse
import yaml
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file',
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


def write_log(config, message):
    with open(os.path.abspath(config['log_file']), 'a') as file:
        file.write(message)
