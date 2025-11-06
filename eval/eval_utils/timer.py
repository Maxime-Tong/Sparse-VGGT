import torch

class CudaTimer:
    def __init__(self, name="Timer", enable=True, stream=None):
        self.name = name
        self.enable = enable
        self.stream = stream
        self.start_event = None
        self.end_event = None

    def start(self):
        if not self.enable:
            return
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record(stream=self.stream)

    def stop(self):
        if not self.enable:
            return
        self.end_event.record(stream=self.stream)
        torch.cuda.synchronize(self.stream)
        elapsed = self.start_event.elapsed_time(self.end_event)
        return elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = self.stop()
        print(f"[{self.name}] Elapsed: {elapsed/1000:.3f} [s]")