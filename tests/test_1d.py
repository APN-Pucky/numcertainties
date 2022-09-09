import pytest
from numcertainties import jacobian_uncertainty
from numcertainties import semi_analytic_uncertainty
from numcertainties import monte_carlo_uncertainty

def test_pow():
	x = [1, 2]
	xcov = [[3, 1],
	        [1, 4]]
	for unc in jacobian_uncertainty,semi_analytic_uncertainty:
		n = unc(x, xcov)
		nn = (n**4).propagate()
		print (nn.get_value(), nn.get_std())
		assert nn.get_value() == pytest.approx([1., 16.])
		assert nn.get_std() == pytest.approx([6.928203230275509,64.])

	x = [1, 2]
	xcov = [[0.03**2, 0],
	        [0, 0.04**2]]
	for unc in jacobian_uncertainty,semi_analytic_uncertainty,monte_carlo_uncertainty:
		n = unc(x, xcov)
		nn = (n**2).propagate()
		print (nn.get_value(), nn.get_std())
		assert nn.get_value() == pytest.approx([1., 4.], rel=1e-3 if unc is monte_carlo_uncertainty else 0)
		assert nn.get_std() == pytest.approx([0.06,0.16], rel=1e-3 if unc is monte_carlo_uncertainty else 0)

test_pow()
	