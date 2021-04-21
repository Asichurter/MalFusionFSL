import time


class StepTimer:
    def __init__(self, total_steps=None):
        self.Lauched = False
        self.LastStep = None
        self.BeginStep = None
        self.TotalSteps = total_steps
        self.CurrentStep = 0
        self.TimeFormatter = TimeFormatter()

        self.CachedData = {}

    def begin(self):
        self.Lauched = True
        self.BeginStep = self.LastStep = time.time()

    def step(self, step_stride, prt=True, end=False):
        if not self.Lauched:
            raise ValueError("[StepTimer] Timer has not be lauched!")

        self.CurrentStep += step_stride
        now_time = time.time()

        cycle_time = now_time- self.LastStep
        cycle_remained = (self.TotalSteps - self.CurrentStep) / step_stride

        if prt:
            print("Time consuming: %.2f" % cycle_time)
            if self.TotalSteps is not None:
                remaining_time = cycle_time * cycle_remained
                # remaining_hour = remaining_time // 3600
                # remaining_min = (remaining_time % 3600) // 60
                # remaining_sec = (remaining_time % 60)
                formatted_time_str = self.TimeFormatter.format(remaining_time)
                print('time remains:', formatted_time_str)
                # print("*"*50)

        self.LastStep = now_time

        if end:
            all_time = time.time() - self.BeginStep
            self.CachedData['total_time'] = all_time
            self.CachedData['avg_time_per_step'] = all_time / self.TotalSteps

    def getTotalTimeStat(self, stat_type='total'):
        if stat_type == 'total':
            return self.CachedData['total_time']
        elif stat_type == 'avg':
            return self.CachedData['avg_time_per_step']
        else:
            raise ValueError(f"[StepTimer] Unrecognized stat type: {stat_type}")


class TimeFormatter:
    def __init__(self, fmt_str='%02d:%02d:%02d'):
        self.FormatString = fmt_str

    @staticmethod
    def _parse(total_second):
        hour = total_second // 3600
        minute = (total_second % 3600) // 60
        second = total_second % 60

        return hour, minute, second

    def format(self, total_second):
        h, m, s = self._parse(total_second)
        return self.FormatString % (h, m, s)
