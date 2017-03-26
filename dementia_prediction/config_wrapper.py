"""
This modules allows to access the yaml config file as if
it was a python module. The config is yaml since sumatra
does not provide an easy and fast way to allow for
python modules as parameter files.
"""
import importlib
import yaml


class Config:
    """ Wrapper class for the config.
    Before first use call parse_config_file
    or you will get an empty config object"""

    # Stores the parsed config
    config = {}

    @staticmethod
    def parse(filename):
        """ Read and parse yaml config file, initialized Config.config.
        Parses a yaml config file and returns a ConfigWrapper object
        with the attributes from the config file but with classes
        instead of strings as values.
        """
        with open(filename) as config_file:
            config_dict = yaml.load(config_file)
        Config.import_python_classes(config_dict)

        Config.config = config_dict

    @staticmethod
    def import_python_classes(obj):
        """ Replace 'module' and 'class' attributes with python objects """

        # Do the wrapper magic only if there is a 'module'
        # and a 'class' attribute(and obviously is dict)
        if isinstance(obj, dict):
            if 'module' in obj and 'class' in obj:
                # Assign obj['module'] to the python module
                # instead of the string
                obj['module'] = importlib.import_module(obj['module'])

                # Assign obj['class'] to the python class instead of the string
                obj['class'] = getattr(obj['module'], obj['class'])

            # Do the same thing for all other keys
            for key, value in obj.items():
                if key != 'module' and key != 'class':
                    Config.import_python_classes(value)

        # If the object is a list, continue the search for each item
        if isinstance(obj, list):
            for item in obj:
                Config.import_python_classes(item)
