import sys
# import your modules and packages from anywhere, i.e., from any directory on your computer.
sys.path.append('../utils/utils')

from custom_logger import get_custom_logger

logger = get_custom_logger(__name__)

# example usage
logger.debug('Debug message')
logger.info('Info message')
logger.warning('Warning message')
logger.error('Error message')
logger.critical('Critical message')