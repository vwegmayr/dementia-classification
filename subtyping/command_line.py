""" provide command line functionality """
from PROJECT.config_wrapper import Config


def main():
    """ initiate PROJECT """
    Config.parse_config_file('example_config.yaml')

    print('Nothing to see here.')
