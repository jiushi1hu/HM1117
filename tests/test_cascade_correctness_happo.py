"""HAPPO correctness observers are independent, bounded and observational."""
import copy
import json
from pathlib import Path
import random
import sys
import unittest

import numpy as np
import torch
from gym.spaces import Box, Discrete

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harl.algorithms.actors.happo import HAPPO
from harl.algorithms.critics.v_critic import VCritic
from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from harl.common.valuenorm import ValueNorm
from harl.runners.on_policy_ha_runner import OnPolicyHARunner


def fixture(fixed_order=False):
    args = dict(hidden_sizes=[8, 8], activation_func='relu',
        use_feature_normalization=False, initialization_method='orthogonal_', gain=.01,
        lr=.01, critic_lr=.003, opti_eps=1e-5, weight_decay=0., clip_param=.2,
        ppo_epoch=2, critic_epoch=2, actor_num_mini_batch=2, critic_num_mini_batch=2,
        entropy_coef=.001, use_max_grad_norm=True, max_grad_norm=.5,
        use_policy_active_masks=True, action_aggregation='prod', normalize_shared_advantages=True,
        gamma=.995, gae_lambda=.95, use_gae=True, use_proper_time_limits=True,
        critic_output_dim=1, value_loss_coef=1., use_clipped_value_loss=False,
        use_huber_loss=False, huber_delta=10.)
    torch.manual_seed(38)
    runner = OnPolicyHARunner()
    runner.num_agents, runner.fixed_order, runner.action_aggregation = 5, fixed_order, 'prod'
    runner.update_count = 3
    runner.correctness_enabled = False
    runner.algo_args = dict(train=dict(episode_length=4, n_rollout_threads=2),
                            algo=dict(mask_inactive_factors=True))
    observation = Box(-np.inf, np.inf, (2,), dtype=np.float32)
    state = Box(-np.inf, np.inf, (3,), dtype=np.float32)
    runner.actor = [HAPPO(args, observation, Discrete(3)) for _ in range(5)]
    runner.actor_buffer = [OnPolicyActorBuffer(runner.algo_args['train'], observation, Discrete(3))
                           for _ in range(5)]
    for index, (actor, buffer) in enumerate(zip(runner.actor, runner.actor_buffer)):
        buffer.obs[:-1] = (np.arange(16, dtype=np.float32).reshape(4, 2, 2)-8.)/7.+index/5.
        buffer.actions[..., 0] = (np.arange(8).reshape(4, 2)+index) % 3
        if index == 2:
            buffer.active_masks[:-1] = 0.
        else:
            buffer.active_masks[index % 4, index % 2] = 0.
        with torch.no_grad():
            logp, _, _ = actor.evaluate_actions(buffer.obs[:-1].reshape(-1, 2),
                                                buffer.actions.reshape(-1, 1))
        buffer.action_log_probs[:] = np.where(buffer.active_masks[:-1] > 0.,
                                              logp.numpy().reshape(4, 2, 1), 0.)
    runner.critic_buffer = OnPolicyCriticBufferEP(dict(args, **runner.algo_args['train']), state)
    runner.critic_buffer.share_obs[:-1] = np.arange(24, dtype=np.float32).reshape(4, 2, 3)/17.
    runner.critic_buffer.returns[:-1, :, 0] = np.array([[-3., 2.], [4., -1.], [5., -4.], [1., 3.]])
    runner.critic_buffer.masks[-1] = 0.
    runner.critic = VCritic(args, state)
    runner.value_normalizer = ValueNorm(1)
    runner.value_normalizer.update(np.array([[-2.], [0.], [3.]], dtype=np.float32))
    return runner


def probabilities(actor, buffer):
    """Use full-distribution probabilities rather than the observer's log sum."""
    with torch.no_grad():
        _, _, distribution = actor.evaluate_actions(buffer.obs[:-1].reshape(-1, 2),
                                                     buffer.actions.reshape(-1, 1))
    return distribution.probs.detach().numpy().astype(np.float64)


class HAPPOCorrectnessObserverTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def assert_state_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for name in left:
                self.assert_state_equal(left[name], right[name])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for x, y in zip(left, right):
                self.assert_state_equal(x, y)
        else:
            self.assertEqual(left, right)

    def test_opt_in_observer_preserves_optimizer_models_normalizer_and_rng_exactly(self):
        plain = fixture()
        measured = copy.deepcopy(plain)
        measured.correctness_enabled = True
        random.seed(52)
        np.random.seed(52)
        python_state, numpy_state = random.getstate(), np.random.get_state()
        torch.manual_seed(52)
        plain_info = plain.train()
        torch_after_plain = torch.get_rng_state().clone()
        torch.manual_seed(52)
        measured_info = measured.train()
        self.assertTrue(torch.equal(torch_after_plain, torch.get_rng_state()))
        self.assert_state_equal(python_state, random.getstate())
        self.assert_state_equal(numpy_state, np.random.get_state())
        self.assert_state_equal(plain_info, measured_info)
        for left, right in zip(plain.actor, measured.actor):
            self.assert_state_equal(left.actor.state_dict(), right.actor.state_dict())
            self.assert_state_equal(left.actor_optimizer.state_dict(), right.actor_optimizer.state_dict())
        self.assert_state_equal(plain.critic.critic.state_dict(), measured.critic.critic.state_dict())
        self.assert_state_equal(plain.critic.critic_optimizer.state_dict(),
                                measured.critic.critic_optimizer.state_dict())
        self.assert_state_equal(plain.value_normalizer.state_dict(), measured.value_normalizer.state_dict())
        for left, right in zip(plain.actor_buffer, measured.actor_buffer):
            self.assert_state_equal(left.factor, right.factor)
        self.assertFalse(hasattr(plain, 'last_correctness_happo'))
        self.assertEqual(measured.last_correctness_happo['update'], 4)
        json.dumps(measured.last_correctness_happo, allow_nan=False)

    def test_factor_summary_matches_direct_probability_product_with_forced_rows(self):
        runner = fixture(fixed_order=True)
        runner.correctness_enabled = True
        old = [probabilities(actor, buffer) for actor, buffer in zip(runner.actor, runner.actor_buffer)]
        torch.manual_seed(17)
        runner.train()
        report = runner.last_correctness_happo
        self.assertEqual(report['agent_order'], list(range(5)))
        prefix = np.ones((4, 2, 1), dtype=np.float64)
        nonunit_seen = False
        for position, row in enumerate(report['actors']):
            agent_id = row['agent_id']
            buffer = runner.actor_buffer[agent_id]
            self.assertEqual(row['update_position'], position)
            np.testing.assert_allclose(buffer.factor, prefix, rtol=1e-6, atol=1e-6)
            for name, expected in (('min', prefix.min()), ('median', np.median(prefix)),
                                   ('p95', np.percentile(prefix, 95)), ('max', prefix.max())):
                self.assertAlmostEqual(row['expected_factor'][name], float(expected), places=6)
            self.assertLess(row['factor_max_relative_error'], 1e-5)
            nonunit_seen |= not np.allclose(prefix, 1., atol=1e-4)
            active = buffer.active_masks[:-1] > 0.
            self.assertEqual(row['forced_rows'], int((~active).sum()))
            self.assertEqual(row['controllable_rows'], int(active.sum()))
            self.assertEqual(row['forced_factor_ratio_max_abs_error'], 0.)
            new = probabilities(runner.actor[agent_id], buffer)
            commands = buffer.actions.reshape(-1).astype(np.int64)
            indices = np.arange(commands.size)
            ratio = (new[indices, commands]/old[agent_id][indices, commands]).reshape(prefix.shape)
            prefix *= np.where(active, ratio, 1.)
        self.assertTrue(nonunit_seen)

    def test_random_order_is_actual_actor_entry_order_and_empty_actor_is_explicit(self):
        runner = fixture()
        runner.correctness_enabled = True
        calls = []
        for agent_id, actor in enumerate(runner.actor):
            train = actor.train
            def observed(*args, _agent_id=agent_id, _train=train, **kwargs):
                calls.append(_agent_id)
                return _train(*args, **kwargs)
            actor.train = observed
        torch.manual_seed(13)
        runner.train()
        report = runner.last_correctness_happo
        self.assertEqual(report['agent_order'], calls)
        self.assertEqual(sorted(calls), list(range(5)))
        self.assertEqual([row['agent_id'] for row in report['actors']], calls)
        forced = next(row for row in report['actors'] if row['agent_id'] == 2)
        self.assertEqual(forced['final_sampled_ratio'],
                         dict(count=0, min=None, median=None, p95=None, max=None))
        self.assertIsNone(forced['final_policy_kl'])
        self.assertIsNone(forced['final_entropy'])
        self.assertEqual(forced['final_masked_ratio']['min'], 1.)
        self.assertEqual(forced['final_masked_ratio']['max'], 1.)


if __name__ == '__main__':
    unittest.main()
