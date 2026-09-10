import unittest
import numpy as np
from cpmg_ml.noise_statistics import shared_reference_covariance, oracle_separation


class NoiseTests(unittest.TestCase):
    def test_covariance_against_intensity_monte_carlo(self):
        rng = np.random.default_rng(123)
        r2, t, sigma = np.array([4., 12., 25.]), .04, .0001
        n = 200000
        ref = 1 + sigma * rng.standard_normal((n, 1))
        intensity = np.exp(-r2*t) + sigma*rng.standard_normal((n, 3))
        measured = -np.log(intensity/ref)/t
        np.testing.assert_allclose(np.cov(measured.T), shared_reference_covariance(r2,t,sigma), rtol=.025)

    def test_repeated_equal_points_shared_reference(self):
        # Shared-reference uncertainty prevents sqrt(N) growth indefinitely.
        for n in (1, 3, 10):
            covariance = shared_reference_covariance(np.zeros(n), 1., 1.)
            self.assertAlmostEqual(oracle_separation(np.ones(n), covariance), np.sqrt(n/(n+1)))


if __name__ == '__main__':
    unittest.main()
