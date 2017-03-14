""" provide command line functionality """
from subtyping.config_wrapper import Config


def main():
    """ initiate subtyping """
    Config.parse_config_file('example_config.yaml')

    print('Nothing to see here.')
