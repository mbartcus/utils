import sys
# import your modules and packages from anywhere, i.e., from any directory on your computer.
sys.path.append('../utils/utils')

import unittest
import logging
import custom_logger

class TestCustomLogger(unittest.TestCase):
    '''
        In this test, we import the unittest module and the get_custom_logger function from the custom_logger module. 
        We define a TestCustomLogger class that inherits from unittest.TestCase. 
        In the setUp method of this class, we create an instance of the logger using the get_custom_logger function.
        We then define four test methods, one for each of the aspects of the logger that we want to test. The test_logger_exists method checks that the get_custom_logger function returns a logging.Logger object. The test_logger_has_handlers method checks that the logger has at least one handler. The test_logger_console_handler_exists method checks that there is at least one logging.StreamHandler object among the handlers, and the test_logger_file_handler_exists method checks that there is at least one logging.FileHandler object among the handlers.
    '''
    def setUp(self):
        self.logger = custom_logger.get_custom_logger()

    def test_logger_exists(self):
        self.assertIsInstance(self.logger, logging.Logger)

    def test_logger_has_handlers(self):
        self.assertTrue(self.logger.hasHandlers())

    def test_logger_console_handler_exists(self):
        handlers = self.logger.handlers
        console_handler = [h for h in handlers if isinstance(h, logging.StreamHandler)]
        self.assertTrue(len(console_handler) > 0)

    def test_logger_file_handler_exists(self):
        handlers = self.logger.handlers
        file_handler = [h for h in handlers if isinstance(h, logging.FileHandler)]
        self.assertTrue(len(file_handler) > 0)

if __name__ == '__main__':
    unittest.main()


'''
#To use it in future in our code we do it as follows:

from custom_logger import get_custom_logger

logger = get_custom_logger(__name__)

# example usage
logger.debug('Debug message')
logger.info('Info message')
logger.warning('Warning message')
logger.error('Error message')
logger.critical('Critical message')
'''