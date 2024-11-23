import os
import pytz

import datetime
import logging
from logging.handlers import TimedRotatingFileHandler


class LogModule:
    def __init__(self, logger_name: str, file_path: str, flag="DEBUG", time_zone="local", today:str=None):
        self.file_path = file_path
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok=True, mode=0o777)

        ## time rotation
        if today is None:
            today = datetime.datetime.now().strftime('%Y-%m-%d')
        else:
            pass
        file_path = file_path.replace('.log', f'_{today}.log')

        tz_map = {"local": 'Asia/Taipei', "utc": "UTC"}
        tz = pytz.timezone(tz_map[time_zone])
        self.log_format = CustomFormatter(fmt='%(asctime)s, %(name)s, %(levelname)s, %(message)s', datefmt='%Y-%m-%d %H:%M:%S %Z', tz=tz)

        self.log_handler = logging.FileHandler(file_path)

        self.log_handler.setFormatter(self.log_format)
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(getattr(logging, flag.upper()))
        self.logger.addHandler(self.log_handler)
        
        
    def rm_handler(self):
        self.logger.removeHandler(self.log_handler)

class CustomFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, tz=None):
        super().__init__(fmt, datefmt)
        self.tz = tz

    def formatTime(self, record, datefmt=None):
        ct = datetime.datetime.fromtimestamp(record.created, self.tz)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            t = ct.strftime("%Y-%m-%d %H:%M:%S")
            s = "%s,%03d" % (t, record.msecs)
        return s

class CustomLog:
    def __init__(self, log_path, logger_name="ty_END", time_zone:str='local', today:str=None) -> None:
        self.log = LogModule(logger_name, log_path, "DEBUG", time_zone=time_zone, today=today)
        #self.log = LogModule(logger_name, log_path, "INFO", time_zone=time_zone, today=today)
        self.logger = self.log.logger
        self.logged_messages = set()

    def make_log(self, msg, level="info", show: bool = True):
        if msg in self.logged_messages:
            return
            
        log_method = getattr(self.logger, level, None)
        if log_method is not None:
            log_method(msg)
            if show:
                # Create a new logger
                print_logger = logging.getLogger('print_logger')
                print_logger.setLevel(getattr(logging, level.upper()))

                # Add a StreamHandler to it
                if not print_logger.handlers:
                    print_logger.addHandler(logging.StreamHandler())

                # Log a message
                getattr(print_logger, level.lower())(msg)

        else:
            self.logger.error(f"Invalid log level: {level}")


if __name__ == "__main__":
    import time
    # Create a logger

    for day in ['2024-06-25', '2024-06-26', '2024-06-27']:
        logger = CustomLog('data/log_test/log/log_tester.log', 'my_logger_test', today=day)
        for j in range(100):
            logger.make_log(f'This is a debug message number {day}-{j}', 'info', show=True)
            # time.sleep(0.2)
        # time.sleep(60)
    # # Create a logger
    # logger = CustomLog('log/log_tester.log' , 'my_logger_test')

    # # Test the logger
    # logger.make_log('This is a debug message', 'debug', show=True)
    # logger.make_log('This is an info message', 'info', show=True)
    # logger.make_log('This is a warning message', 'warning', show=False)
    # logger.make_log('This is an error message', 'error', show=False)
    # logger.make_log('This is a critical message', 'critical', show=False)
    # # Create a logger
    # logger = CustomLog('log/log_tester_utc.log' , 'my_logger_test', 'utc')

    # # Test the logger
    # logger.make_log('This is a debug message', 'debug', show=True)
    # logger.make_log('This is an info message', 'info', show=True)
    # logger.make_log('This is a warning message', 'warning', show=False)
    # logger.make_log('This is an error message', 'error', show=False)
    # logger.make_log('This is a critical message', 'critical', show=True)