

class Trial_data:
    def __init__(self, succ, duration, j0s, j1s, j0e, j1e):
        self.succ = succ
        self.duration = duration
        self.j0s = j0s
        self.j1s = j1s
        self.j0e = j0e
        self.j1e = j1e


class SAR_point:
    def __init__(self, trial_id, step, j0c, j1c, a0, a1, r):
        self.trial_id = trial_id
        self.step = step
        self.j0c = j0c
        self.j1c = j1c
        self.a0 = a0
        self.a1 = a1
        self.reward = r

class Saqt_point:
    def __init__(self, id, j0s, j1s, v0s, v1s, j0t, j1t, a0, a1, qt):
        self.id = id
        self.j0s = j0s
        self.j1s = j1s
        self.v0s = v0s
        self.v1s = v1s
        self.j0t = j0t
        self.j1t = j1t
        self.a0 = a0
        self.a1 = a1
        self.qt = qt
