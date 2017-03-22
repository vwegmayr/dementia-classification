""" provide command line functionality """
import argparse
from .config_wrapper import Config


def main():
    """Command Line Interface"""

    parser = argparse.ArgumentParser(
        description='Description of your commandline tool'
    )

    parser.add_argument(
        "configfile",
        type=str,
        help="Path to configuration file"
    )

    args = parser.parse_args()

    Config.parse(args.configfile)

    print("======================================")
    print("Take a look at the example config.yml:")
    print("======================================")
    print(Config.config)
    print("======================================")