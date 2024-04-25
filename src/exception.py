import sys
from src.logger import logging


def error_message_details(message, error_details: sys):
    _, _, exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in python script [{file_name}]; line number [{exc_tb.tb_lineno}]; error message [{message}]"

    return error_message


class CustomException(Exception):

    def __init__(self, message, details):
        super().__init__(message)
        self.message = error_message_details(message, details)

    def __str__(self):
        return self.message
 