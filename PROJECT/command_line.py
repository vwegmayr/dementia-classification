""" provide command line functionality """
import importlib
from settings import PROJECT
CONFIG_WRAPPER = importlib.import_module(PROJECT + ".config_wrapper")
CONFIG = CONFIG_WRAPPER.Config


def main():
    """Command Line Interface"""
    CONFIG.parse_config_file(PROJECT + "/examples/example_config.yaml")

    print('Nothing to see here.')
