import numpy as np
import time

class Accumulator:
    def __init__(self, n) -> None:
        self.data = [0.0]*n

    def add(self, *args):
        if len(args) == len(self.data):
            self.data = [a + float(b)
                         for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Timer:
    """Record multiple running times"""

    def __init__(self) -> None:
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times)/len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()
