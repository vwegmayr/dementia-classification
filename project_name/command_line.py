""" provide command line functionality """
from project_name.config_wrapper import Config


def main():
    """ initiate project_name """
    Config.parse_config_file('example_config.yaml')

    print('Nothing to see here.')
