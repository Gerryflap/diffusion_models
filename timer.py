import time


class Timer:
    """
        A small utility class that can be used to track the duration of certain operations
    """
    def __init__(self, enabled=True):
        """
            Instantiates the timer.
            "enabled" parameter can be used to easily disable the timer in one place when you want to disable it
            without deleting all the log and sets from the code.
        """
        self.t = 0
        self.enabled = enabled

    def set(self):
        """
            Sets the internal timer to now. Should be done at the start of whatever you want to time.
        """
        if not self.enabled:
            return
        self.t = time.time()

    def log_and_set(self, string: str):
        """
            Convenience method to call log and set in one go
        """
        if not self.enabled:
            return
        self.log(string)
        self.set()

    def log(self, string: str):
        """
            Prints the time delta in milliseconds since the previous .set().
            Does not re-set the timer.
        """
        if not self.enabled:
            return
        t_new = time.time()
        t_diff = t_new - self.t
        print("%s took: %.1f ms" % (string, t_diff * 1000))
