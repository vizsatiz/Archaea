import logging.config
import settings as settings

# Using the config file for console logging
logging.config.fileConfig(settings.PROJECT_ROOT + '/log.conf')
# Creating handler for file logging
handler = logging.FileHandler(settings.PROJECT_ROOT + '/bin/logs/info.log')
# Format for file logging
formatter = logging.Formatter('[%(asctime)s][%(threadName)s][%(name)-12s][%(levelname)-2s] : \n \r \t %(message)s \n')
handler.setFormatter(formatter)
# create logger for data adapter
logger = logging.getLogger('archaea')
# assinging the handler
logger.addHandler(handler)

# Creating another handler for error logging which logs to error file
error_handler = logging.FileHandler(settings.PROJECT_ROOT + '/bin/logs/error.log')
error_formatter = logging.Formatter('[%(asctime)s][%(threadName)s][%(name)-12s][%(levelname)-2s] : \n \r \t %(message)s \n')
error_handler.setFormatter(error_formatter)
log_error = logging.getLogger('archaea-error')
log_error.addHandler(error_handler)

# Making the error logger part of logger
logger.error = log_error.error