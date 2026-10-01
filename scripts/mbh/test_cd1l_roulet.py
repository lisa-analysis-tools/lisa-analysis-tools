"""CPU-only checks: python -m unittest discover -s scripts/mbh -p test_cd1l_roulet.py."""
import unittest
import numpy as np
from cd1l_roulet import to_roulet, from_roulet, forward_jacobian, transport_covariance, propose


class SpinCoordinatesTest(unittest.TestCase):
    def test_regression_short_chains_and_bases(self):
        # Exercise the actual driver block without importing its GPU/data setup.
        import ast, contextlib, io, os
        from pathlib import Path
        from unittest.mock import patch
        import cd1l_sampling as sampling
        tree = ast.parse(Path(__file__).with_name('cd1l_pe.py').read_text())
        block = next(n for n in tree.body if isinstance(n, ast.If)
                     and ast.unparse(n.test) == "os.environ.get('CD1L_REGRESS')")
        code = compile(ast.Module(body=[block], type_ignores=[]), '<regression>', 'exec')
        x = np.zeros((2, 11)); x[:, 0] = 12.; x[:, 1] = .4
        for basis in ('mc', 'stock'):
            stored = x if basis == 'mc' else sampling.mc_to_stock(x)
            class Backend:
                def get_chain(self):
                    return {'mbh': np.tile(stored[None, None, :, None, :], (3, 1, 1, 1, 1))}
                def get_log_like(self):
                    return np.zeros((3, 1, 2))
            seen = []
            def likelihood(rows, **kw):
                np.testing.assert_allclose(rows, x, atol=1e-12)
                seen.append(rows.copy())
                return np.zeros(2)
            scope = dict(os=os, np=np, HDFBackend=lambda p: Backend(), SAMPLER='mc',
                         S=sampling, like_fn=likelihood, LIKE_KW={})
            with patch.dict(os.environ, {'CD1L_REGRESS': 'short.h5', 'CD1L_REGRESS_BASIS': basis}), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    exec(code, scope)
                self.assertEqual(raised.exception.code, 0)
            self.assertEqual(len(seen), 3)

    def test_joint_mass_prior(self):
        from cd1l_sampling import MassConstrainedPrior, mtot_ok
        from eryn.priors.analytical import UniformDistribution
        boxes = {j: UniformDistribution(-1., 1.) for j in range(11)}
        boxes[0] = UniformDistribution(np.log(3e3), np.log(4e7))
        boxes[1] = UniformDistribution(.05, 1.)
        prior = MassConstrainedPrior(boxes)
        x = np.zeros((2, 11)); x[:, 0] = np.log(4e7) - 1e-6
        x[:, 1] = [.051, .999]
        self.assertFalse(np.isfinite(prior.logpdf(x)[0]))
        self.assertTrue(np.isfinite(prior.logpdf(x)[1]))
        self.assertTrue(mtot_ok(prior.rvs(size=(4, 100)).reshape(-1, 11)).all())
        # Independent midpoint quadrature of the retained mass-box fraction.
        q = np.linspace(.05, 1., 100001)
        upper = np.log(1e8) + .6*np.log(q) - 1.2*np.log1p(q)
        fraction = np.mean(np.clip((upper-np.log(3e3))/np.log(4e7/3e3), 0, 1))
        self.assertAlmostEqual(np.exp(prior.log_fraction), fraction, places=5)

    def test_window_grid_invariance(self):
        from cd1l_windows import observation_window
        full = observation_window(600, 1000, 2.5, 250., 100.)
        short = observation_window(600, 640, 2.5, 250., 100.)
        np.testing.assert_array_equal(full[:640], short)
        self.assertTrue((full[600:] == 0).all())
        self.assertEqual(full[0], 0.)
        self.assertEqual(full[599], 0.)
        # Entry ramp agrees at identical physical sample times after decimation.
        dec = observation_window(150, 160, 10., 250., 100.)
        np.testing.assert_allclose(full[:400:4], dec[:100])

    def setUp(self):
        self.rng = np.random.RandomState(71)
        self.x = self.rng.uniform(-.8, .8, (100, 11))
        self.x[:, 1] = self.rng.uniform(.05, 1, 100)

    def test_roundtrip(self):
        np.testing.assert_allclose(from_roulet(to_roulet(self.x)), self.x, atol=1e-15)
        self.assertTrue(np.array_equal(to_roulet(self.x)[:, 4:], self.x[:, 4:]))

    def test_full_jacobian(self):
        for x in self.x[:10]:
            h = np.eye(11) * 1e-6
            numerical = ((to_roulet(x + h) - to_roulet(x - h)) / 2e-6).T
            np.testing.assert_allclose(forward_jacobian(x), numerical, atol=1e-9)
            self.assertAlmostEqual(abs(np.linalg.det(numerical)), 1., places=8)

    def test_covariance(self):
        a = self.rng.normal(size=(11, 11))
        cov = a @ a.T + np.eye(11)
        j = forward_jacobian(self.x[0])
        transported = transport_covariance(cov, self.x[0])
        np.testing.assert_allclose(transported, j @ cov @ j.T, atol=1e-13)
        np.linalg.cholesky(transported)

    def test_rejection_and_wrap(self):
        x = self.x.copy()
        x[:, 5] %= 2*np.pi
        lo, hi = np.full(11, -1.), np.full(11, 1.)
        lo[1], hi[1] = .05, 1.
        lo[5], hi[5] = 0., 2*np.pi
        steps = np.zeros_like(x)
        steps[0, 1] = -10
        steps[1, 2] = 10
        steps[2, 5] = 2*np.pi + .01
        out, valid = propose(x, steps, lo, hi, {5: 2*np.pi})
        self.assertFalse(valid[0])
        self.assertFalse(valid[1])
        np.testing.assert_array_equal(out[:2], x[:2])
        self.assertTrue(valid[2])
        self.assertAlmostEqual(out[2, 5], (x[2, 5]+.01) % (2*np.pi))

    def test_eryn_move(self):
        from cd1l_sampling import RouletSpinGaussianMove
        lo, hi = np.full(11, -100.), np.full(11, 100.)
        lo[1] = .001
        move = RouletSpinGaussianMove(np.eye(11)*1e-5, self.x[0], lo, hi)
        coords = self.x[:12].reshape(2, 6, 1, 11)
        result, factors = move.get_proposal({'mbh': coords}, self.rng)
        self.assertEqual(result['mbh'].shape, coords.shape)
        np.testing.assert_array_equal(factors, np.zeros((2, 6)))
        self.assertEqual(move.attempted, 12)
        self.assertEqual(move.prior_valid, 12)


if __name__ == '__main__':
    unittest.main()
