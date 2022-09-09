import pytest
import numpy as np
from numcertainties import jacobian_uncertainty
from numcertainties import semi_analytic_uncertainty
from numcertainties import monte_carlo_uncertainty

def test_pow():
	x = 1
	xcov = 0.5
	for unc in jacobian_uncertainty,semi_analytic_uncertainty,monte_carlo_uncertainty:
		n = unc(x, xcov)
		nn = (n+2).propagate()
		print (nn.get_value(), nn.get_std())
		assert nn.get_value() == pytest.approx(3., rel=1e-3 if unc is monte_carlo_uncertainty else 0)
		assert nn.get_std() == pytest.approx(0.5**0.5, rel=1e-3 if unc is monte_carlo_uncertainty else 0)

	x = 10
	xcov = 1e-4**2
	for unc in semi_analytic_uncertainty,jacobian_uncertainty,monte_carlo_uncertainty:
		n = unc(x, xcov)
		nn = (n**2).propagate()
		print (nn.get_value(), nn.get_std())
		assert nn.get_value() == pytest.approx(100, rel=1e-3 if unc is monte_carlo_uncertainty else 0)
		assert nn.get_std() == pytest.approx(0.002, rel=1e-3 if unc is monte_carlo_uncertainty else 0)

def test_mult_pow():
	x = 10
	xcov = 1e-4**2
	for unc in semi_analytic_uncertainty,jacobian_uncertainty,monte_carlo_uncertainty:
		n = unc(x, xcov)
		nn = (n**2).propagate()
		nnn = (n*n).propagate()
		print("mp",nn.get_value(), nn.get_std())
		print("mp",nnn.get_value(), nnn.get_std())
		assert np.allclose(nn.get_value(), nnn.get_value(),rtol=1e-3 if unc is monte_carlo_uncertainty else 0)
		assert np.allclose(nn.get_std(), 	nnn.get_std(),rtol=1e-3 if unc is monte_carlo_uncertainty else 0)


def test_add():
	x = 1
	xcov = 0.5
	for unc in jacobian_uncertainty,semi_analytic_uncertainty,monte_carlo_uncertainty:
		n = unc(x, xcov)
		nn = (n+2).propagate()
		print (nn.get_value(), nn.get_std())
		assert nn.get_value() == pytest.approx(3., rel=1e-3 if unc is monte_carlo_uncertainty else 0)
		assert nn.get_std() == pytest.approx(0.5**0.5, rel=1e-3 if unc is monte_carlo_uncertainty else 0)

	y = 10
	ycov = 1e-4**2
	for unc in semi_analytic_uncertainty,jacobian_uncertainty,monte_carlo_uncertainty:
		n = unc(y, ycov)
		nn = (n+2).propagate()
		print (nn.get_value(), nn.get_std())
		assert nn.get_value() == pytest.approx(12, rel=1e-3 if unc is monte_carlo_uncertainty else 0)
		assert nn.get_std() == pytest.approx(1e-4, rel=1e-3 if unc is monte_carlo_uncertainty else 0)

	for unc in monte_carlo_uncertainty,jacobian_uncertainty,semi_analytic_uncertainty:
		n = unc(x, xcov)
		m = unc(y, ycov)
		nn = (n+m).propagate()
		print (nn.get_value(), nn.get_std())
		assert nn.get_value() == pytest.approx(11, rel=1e-3 if unc is monte_carlo_uncertainty else 0)
		assert nn.get_std() == pytest.approx(np.sqrt(0.5 + 1e-4**2), rel=1e-3 if unc is monte_carlo_uncertainty else 0)

test_add()
	