import sys 

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = f"Error occurred in script: {file_name} at line {line_number} with message: {error}"
    else:
        error_message = f"Error: {error}"
    return error_message

class customException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        _, _, exc_tb = error_detail.exc_info()
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            self.error_message = f"Error occurred in script: {file_name} at line {line_number} with message: {error_message}"
        else:
            self.error_message = f"Error: {error_message}"

    def __str__(self):
        return self.error_message
