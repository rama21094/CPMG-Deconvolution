"""High-SNR, first-order R2 covariance for a shared reference measurement.

This approximation is unsuitable for intensities near/below the noise floor.
An oracle separation statistic does not establish inverse-problem recovery.
"""
import numpy as np


def shared_reference_covariance(r2, time_t2, sigma_rel):
    r2 = np.asarray(r2, dtype=float)
    if time_t2 <= 0 or sigma_rel <= 0 or not np.isfinite(r2).all():
        raise ValueError('Finite rates, positive time and noise are required')
    variance = (sigma_rel / time_t2) ** 2
    return variance * (np.diag(np.exp(2 * r2 * time_t2)) + np.ones((r2.size, r2.size)))


def oracle_separation(difference, covariance):
    d = np.asarray(difference, dtype=float)
    return float(np.sqrt(max(0.0, d @ np.linalg.solve(covariance, d))))
