"""CPU checks for reflection semantics, including arbitrary joint ordering."""
import importlib.util
from pathlib import Path
import unittest
import torch

MODULE = Path(__file__).resolve().parents[1] / 'HIMLoco/legged_gym/legged_gym/utils/lz_q1l_mirror.py'
spec = importlib.util.spec_from_file_location('mirror', MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MirrorTests(unittest.TestCase):
    def setUp(self):
        # Deliberately different from simulator order.
        self.names = [kind + '_leg_' + leg + '_joint'
                      for kind in ('pitch02', 'roll', 'pitch01')
                      for leg in ('b_r', 'f_l', 'b_l', 'f_r')]
        self.mirror = module.LZQ1LMirror(self.names)

    def test_joint_semantics_and_roundtrip(self):
        q = torch.arange(1., 13.)
        actual = self.mirror.joints(q)
        for kind, sign in [('roll', -1), ('pitch01', 1), ('pitch02', 1)]:
            for end in ('f', 'b'):
                left = self.names.index(kind + '_leg_' + end + '_l_joint')
                right = self.names.index(kind + '_leg_' + end + '_r_joint')
                self.assertEqual(actual[left], sign * q[right])
                self.assertEqual(actual[right], sign * q[left])
        self.assertTrue(torch.equal(self.mirror.joints(actual), q))

    def test_all_frames_and_vector_semantics(self):
        obs = torch.arange(1., 541.).reshape(2, 270)
        before = obs.clone()
        actual = self.mirror.history(obs)
        self.assertTrue(torch.equal(obs, before))
        self.assertTrue(torch.equal(self.mirror.history(actual), obs))
        expected_signs = torch.tensor([1., -1., -1., -1., 1., -1., 1., -1., 1.])
        self.assertTrue(torch.equal(actual.reshape(2, 6, 45)[..., :9],
                                    obs.reshape(2, 6, 45)[..., :9] * expected_signs))
        # A right-front roll action must reach the left-front roll, negated,
        # in every historical action block, with no time reversal.
        left = self.names.index('roll_leg_f_l_joint')
        right = self.names.index('roll_leg_f_r_joint')
        for frame in range(6):
            self.assertTrue(torch.equal(actual[:, frame * 45 + 33 + left],
                                        -obs[:, frame * 45 + 33 + right]))

    def test_wrapper_and_estimator(self):
        class Actor:
            def act_inference(self, obs):
                self.seen = obs.clone()
                return obs[:, 33:45]

            def estimator(self, obs):
                self.seen = obs.clone()
                return obs[:, :3], obs[:, 9:25]
        actor = Actor()
        policy = module.LeftMirroredLZQ1LPolicy(actor, self.mirror, [2., 2., .25])
        obs = torch.arange(1., 271.).reshape(1, -1)
        obs[:, :3] = torch.tensor([[0., .8, 0.]])
        self.assertTrue(torch.equal(policy(obs), obs[:, 33:45]))
        self.assertTrue(torch.equal(actor.seen, self.mirror.history(obs)))
        velocity, latent = policy.estimator(obs)
        self.assertTrue(torch.equal(velocity, torch.tensor([[0., .8, 0.]])))
        self.assertTrue(torch.equal(latent, actor.seen[:, 9:25]))

    def test_reject_other_layout(self):
        with self.assertRaises(ValueError):
            module.LZQ1LMirror(['unknown'] * 12)
        with self.assertRaises(ValueError):
            self.mirror.history(torch.zeros(1, 45))

    def test_conditional_selection_matches_each_controller(self):
        class Actor:
            def act_inference(self, obs):
                return obs[:, :12] + torch.arange(12.)

            def estimator(self, obs):
                return obs[:, 3:6] + 1., obs[:, 10:26]
        actor = Actor()
        policy = module.LeftMirroredLZQ1LPolicy(actor, self.mirror, [2., 2., .25])
        obs = torch.arange(1., 6 * 270 + 1).reshape(6, 270)
        # left, right, stand, forward, diagonal-left, left+yaw (scaled commands).
        obs[:, :3] = torch.tensor([[0., .8, 0.], [0., -.8, 0.], [0., 0., 0.],
                                   [1., 0., 0.], [.6, .8, 0.], [0., .8, .2]])
        before = obs.clone()
        self.assertEqual(policy.mirror_mask(obs).tolist(), [True, False, False, False, False, False])
        expected = actor.act_inference(obs)
        expected[0] = self.mirror.joints(actor.act_inference(self.mirror.history(obs)))[0]
        self.assertTrue(torch.equal(policy(obs), expected))
        velocity, latent = policy.estimator(obs)
        expected_v, expected_z = actor.estimator(obs)
        expected_z = expected_z.clone()
        mirrored_v, mirrored_z = actor.estimator(self.mirror.history(obs))
        mirrored_v = self.mirror.linear_velocity(mirrored_v)
        expected_v[0], expected_z[0] = mirrored_v[0], mirrored_z[0]
        self.assertTrue(torch.equal(velocity, expected_v))
        self.assertTrue(torch.equal(latent, expected_z))
        self.assertTrue(torch.equal(obs, before))

    def test_switch_uses_current_command_without_rewriting_history(self):
        policy = module.LeftMirroredLZQ1LPolicy(None, self.mirror, [2., 2., .25])
        obs = torch.arange(1., 271.).reshape(1, 270)
        obs[:, :3] = torch.tensor([[0., .8, 0.]])
        physical_past = obs[:, 45:].clone()
        selected, mask = policy.selected_history(obs)
        self.assertTrue(mask.item())
        self.assertTrue(torch.equal(selected, self.mirror.history(obs)))
        obs[:, 1] = -.8
        selected, mask = policy.selected_history(obs)
        self.assertFalse(mask.item())
        self.assertTrue(torch.equal(selected, obs))
        self.assertTrue(torch.equal(obs[:, 45:], physical_past))


if __name__ == '__main__':
    unittest.main()
