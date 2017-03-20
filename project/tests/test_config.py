""" This module tests the config_wrapper module. """
import unittest
import numbers
import importlib
from settings import PROJECT
CONFIG_WRAPPER = importlib.import_module(PROJECT + ".config_wrapper")
CONFIG = CONFIG_WRAPPER.Config


class TestConfigWrapper(unittest.TestCase):
    """ Test the ConfigWrapper class """

    def setUp(self):
        """ Prepare for tests, so load prepared config. """

        CONFIG.parse_config_file(PROJECT + '/examples/example_config.yaml')

    def test_config_dict_set(self):
        """ Config.config should not be the empty dict anymore """
        self.assertNotEqual(CONFIG.config, {})

    def test_modules_replaced(self):
        """ Test if the string from 'module' attributes correctly
        have been replaced with python modules"""
        params = CONFIG.config['Parameters']

        self.assertEqual(params['module'], numbers)
        self.assertEqual(params['list'][0]['module'], numbers)

    def test_classes_replaced(self):
        """ Test if the string from 'class' attributes correctly
        have been replaced with the corresponding python class """
        params = CONFIG.config['Parameters']

        self.assertEqual(params['class'], numbers.Number)
        self.assertEqual(params['list'][1]['class'], numbers.Number)
