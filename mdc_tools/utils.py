import time

__all__ = ["Timer", ]
class Timer():
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.start = time()

    def __exit__(self, type, value, traceback):
        elapsed_time = (time.time() - self.start)
        print(self.message.format(elapsed_time))

