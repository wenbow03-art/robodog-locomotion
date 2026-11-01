"""Performance-gated acceleration levels; independent of PPO and simulator state."""
class AccelerationCurriculum:
    def __init__(self, cfg):
        self.levels = list(cfg.levels)
        self.minimum = int(cfg.min_stage_updates)
        self.interval = int(cfg.evaluation_interval)
        self.required = int(cfg.required_pass_streak)
        self.fraction = float(cfg.required_pass_fraction)
        self.trials = int(cfg.evaluation_trials)
        self.stage = 0
        self.streak = 0
        self.stage_start = 0
        self.last_evaluation = 0

    def state_dict(self):
        return dict(levels=self.levels, stage=self.stage, streak=self.streak,
                    stage_start=self.stage_start, last_evaluation=self.last_evaluation)

    def load_state_dict(self, state):
        if state['levels'] != self.levels or not 0 <= state['stage'] < len(self.levels):
            raise ValueError('Incompatible acceleration curriculum checkpoint')
        for key in ('stage', 'streak', 'stage_start', 'last_evaluation'):
            setattr(self, key, int(state[key]))

    def consider(self, completed, report):
        if completed-self.last_evaluation < self.interval:
            raise ValueError('Evaluation too early or duplicate')
        self.last_evaluation = completed
        groups = report.get('groups', [])
        expected = self.levels[:self.stage+1]
        passed = (not report.get('error') and [g['acceleration'] for g in groups] == expected)
        if passed:
            passed = all(len(g['trials']) == self.trials and
                         sum(bool(t['passed']) for t in g['trials']) / self.trials >= self.fraction
                         for g in groups)
        eligible = completed-self.stage_start >= self.minimum
        self.streak = min(self.required, self.streak+1) if passed and eligible else 0
        previous = self.stage
        if self.streak >= self.required and self.stage < len(self.levels)-1:
            self.stage += 1
            self.stage_start = completed
            self.streak = 0
        return dict(passed=bool(passed), promoted=self.stage != previous,
                    stage=self.stage, acceleration=self.levels[self.stage], streak=self.streak)
