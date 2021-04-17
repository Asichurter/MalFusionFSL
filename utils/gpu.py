import pynvml

pynvml.nvmlInit()


class GPUManager:
    MemUnitIntMap = {
        'B': 1.,
        'K': 1024.,
        'M': 1024**2,
        'G': 1024**3
    }

    def __init__(self, device_id):
        self.Handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

    def getGPUUsedMem(self, unit='M'):
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.Handle)
        mem_unit_divide_factor = self.MemUnitIntMap[unit]
        return mem_info.used / mem_unit_divide_factor
