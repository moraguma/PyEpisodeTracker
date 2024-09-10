import time
import csv

class Profiller:
    _instance = None

    def __init__(self) -> None:
        self.verbose = True
        
        self.total_times = {"Profiller": 0.0}
        self.last_time_calls = {}
    

    def _start_self(self) -> None:
        self.last_time_calls["Profiller"] = time.time_ns()
    

    def _stop_self(self) -> None:
        self.total_times["Profiller"] = (time.time_ns() - self.last_time_calls["Profiller"]) / 1000000000.0


    def start(self, context: str) -> None:
        self._start_self()
        self.last_time_calls[context] = time.time_ns()
        self._stop_self()
    

    def stop(self, context: str) -> None:
        self._start_self()
        if not context in self.last_time_calls:
            if self.verbose:
                print(f"Profiller: Called stop() on context {context} without calling start() first")
            self._stop_self()
            return

        if not context in self.total_times:
            self.total_times[context] = 0
        
        self.total_times[context] += (time.time_ns() - self.last_time_calls[context]) / 1000000000.0
        self._stop_self()
    

    def save_csv(self, path: str):
        self._start_self()
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, self.total_times.keys())
            w.writeheader()
            w.writerow(self.total_times)
        self._stop_self()


profiller = Profiller()