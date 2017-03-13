""" provide command line functionality """
from parkinson_clustering.config_wrapper import Config


def main():
    """ initiate global tracking """
    Config.parse_config_file('example_config.yaml')

    print('Nothing to see here.')
