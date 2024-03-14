import logging

class Logger():
    def __init__(self) -> None:
        logging.basicConfig(filename='statevec.log', level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    def warning(self, msg):
        logging.warning(msg)

    def error(self, msg):
        logging.error(msg)
    
    def info(self, msg):
        logging.info(msg)

    def debug(self, msg):
        logging.debug(msg)
    
    def exception(self, msg):
        logging.exception(self, msg)
    
logger = Logger()