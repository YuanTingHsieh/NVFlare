import time
from multiprocessing import Lock


class TimestampService:
    lock = Lock()
    timestamps = []

    @staticmethod
    def add_timestamp(description: str):
        with TimestampService.lock:
            TimestampService.timestamps.append((description, time.time()))

    @staticmethod
    def write_timestamps(file_name: str):
        with TimestampService.lock:
            with open(file_name, "w") as f:
                for t in TimestampService.timestamps:
                    f.write(f"{t[0]}:{t[1]}\n")
        TimestampService.timestamps = []
