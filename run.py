"""
    Main caller for operations based on YAML configuration file
"""

from utils import utils
from wikihow import Wikihow
from actionidentifier import ActionIdentifier

def main():
    args = utils.parse_args()

    config = utils.load_config_file(args.config_file)

    # Parse available execution options
    if config['action']['execute'] == 'download_dataset':
        wikihow = Wikihow.Wikihow(config)
        wikihow.download()
    elif config['action']['execute'] == 'action_identifier':
        action_identifier = ActionIdentifier.ActionIdentifier(config)
        action_identifier.run()
    elif config['action']['execute'] == 'dataset_statistics':
        wikihow = Wikihow.Wikihow(config)
        wikihow.get_statistics()
    else:
        print("Invalid execute parameter in config file {}".format(args.config_file))
        exit()



if __name__ == "__main__":
    main()
