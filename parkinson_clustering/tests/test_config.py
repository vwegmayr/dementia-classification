""" This module tests the config_wrapper module. """

import unittest
import yaml
import numpy
from parkinson_clustering.config_wrapper import Config


class TestConfigWrapper(unittest.TestCase):
    """ Test the ConfigWrapper class """

    def setUp(self):
        """ Prepare for tests, so load prepared config. """

        Config.parse_config_file('example_config.yaml')
        """
        config_dict = yaml.load('''
        Parameters:
           module: 'numpy.random'
           class: 'RandomState'
           list:
           - config1:
             weight: 0.2
             module: 'numpy.random'
             class: 'RandomState'
           - config2:
             weight: 0.1
             module: 'numpy.random'
             class: 'RandomState'
             ''')

        Config.import_python_classes(config_dict)
        Config.config = config_dict
        """

    def test_config_dict_set(self):
        """ Config.config should not be the empty dict anymore """
        self.assertNotEqual(Config.config, {})

    def test_modules_replaced(self):
        """ Test if the string from 'module' attributes correctly
        have been replaced with python modules"""
        params = Config.config['Parameters']

        self.assertEqual(params['module'], numpy.random)
        self.assertEqual(params['list'][0]['module'],numpy.random)

    def test_classes_replaced(self):
        """ Test if the string from 'class' attributes correctly
        have been replaced with the corresponding python class """
        params = Config.config['Parameters']

        self.assertEqual(params['class'], numpy.random.RandomState)
        self.assertEqual(params['list'][1]['class'], numpy.random.RandomState)
