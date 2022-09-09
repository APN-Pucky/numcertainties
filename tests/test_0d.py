import pytest
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


test_pow()
	