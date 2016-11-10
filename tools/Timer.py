# Timer.py
import datetime


class Timer:
    time_stack = []

    def start(self):
        self.time_stack.append(datetime.datetime.now())

    def stop(self):
        then = self.time_stack.pop()
        now = datetime.datetime.now()
        return now - then
