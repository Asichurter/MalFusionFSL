import time


class StepTimer:
    def __init__(self, total_steps=None):
        self.Lauched = False
        self.LastStep = None
        self.BeginStep = None
        self.TotalSteps = total_steps
        self.CurrentStep = 0

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
                remaining_hour = remaining_time // 3600
                remaining_min = (remaining_time % 3600) // 60
                remaining_sec = (remaining_time % 60)
                print('time remains:  %02d:%02d:%02d'%(remaining_hour, remaining_min, remaining_sec))
                # print("*"*50)

        self.LastStep = now_time

        if end:
            all_time = time.time() - self.BeginStep
            self.CachedData['total_time'] = all_time
            self.CachedData['avg_time_per_step'] = all_time / self.TotalSteps
            # all_hour = all_time // 3600
            # all_minute = (all_time % 3600) // 60
            # all_second = all_time % 60
            # return "%02d:%02d:%02d"%(all_hour, all_minute, all_second)

    def getTotalTimeStat(self, stat_type='total'):
        if stat_type == 'total':
            return self.CachedData['total_time']
        elif stat_type == 'avg':
            return self.CachedData['avg_time_per_step']
        else:
            raise ValueError(f"[StepTimer] Unrecognized stat type: {stat_type}")
