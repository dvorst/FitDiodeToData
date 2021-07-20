from numpy import exp, real, stack
from numpy import log as ln
from lib.optimizer import Adam
from lib.loss_functions import MSLE
from scipy.special import lambertw


def w(x):
	return real(lambertw(x))


class ShockleyResistanceDiodeModel:
	"""
	Shockley diode model with series resistance
	"""

	def __init__(self, t=25, i_s=1e-14, n=2, r=1):
		self.i_s = i_s
		self.a = ln(i_s)
		self.n = n
		self.r = r
		self.t = t
		self.vt = 25e-3

	@staticmethod
	def derive_equations():
		from sympy import exp, LambertW, symbols, diff, print_latex, simplify
		# define symbolic variables
		a, n, r, vt, vd = symbols('a n r v_t v_d')

		# derive diode current expression
		i_s = exp(a)
		x = i_s * r / (n * vt) * exp((vd + i_s * r) / (n * vt))
		i = simplify(n * vt / r * LambertW(x) - i_s)

		# calculate derivative of diode current in respect to model parameters
		par = [a, n, r]
		di = [simplify(diff(i, p)) for p in par]

		# print result in latex
		print('Id')
		print_latex(i)
		print('')
		print('∂Id/∂a = ')
		print_latex(di[0])
		print('')
		print('∂Id/∂n = ')
		print_latex(di[1])
		print('')
		print('∂Id/∂r = ')
		print_latex(di[2])

	def fit_model_to_data(self, ids, vds, loss=None, optimizer=None):
		"""
		Fit diode model to the data samples of the diode_current (Ids) and
		diode voltage (Vds)
		:param ids: diode current samples
		:param vds: diode voltage samples
		:param loss: loss object, MSLE is used as default
		:param optimizer: optimizer object, Adam is used as default
		:return:
		"""
		loss = MSLE(ids, 1e-12) if loss is None else loss
		optimizer = Adam() if optimizer is None else optimizer

		for idx in range(0, optimizer.max_iter):
			# calculate models prediction (diode current)
			wx = w(self._x(vds))
			ide = self.diode_current(vds, _wx=wx)

			# calculate loss from the models prediction
			l = loss.calculate(ide)

			# return if loss is below threshold
			if l < loss.thr:
				return

			# calculate gradient of loss in respect to models parameters
			d_ide = self.grad_diode_current(vds, wx)
			dl = loss.grad_loss(ide, d_ide)

			# calculate parameter step
			d_par = optimizer.step(dl)

			if idx % 1000 == 0:
				print(
					'%.6i  | %.2e %7.3f %7.3f  | % .0e % .0e % .0e  |  '
					'%.12f' %
					(
						idx, self.a, self.n, self.r,
						d_par[0], d_par[1], d_par[2], l)
				)

			# update parameters
			self.a = self.a + d_par[0]
			self.n = self.n + d_par[1]
			self.r = self.r + d_par[2]

		self.i_s = exp(self.a)

	def diode_current(self, vd, _wx=None):
		"""
		Returns diode current for a given diode voltage
		:param vd: diode voltage [V]
		:param _wx: W(x), reduces number of calculations if provided
		:return: dioe current [A]
		"""
		a, n, r, vt = self.a, self.n, self.r, self.vt
		wx = w(self._x(vd)) if _wx is None else _wx
		return n * vt * wx / r - exp(a)

	def _x(self, vd):
		a, n, r, vt = self.a, self.n, self.r, self.vt
		return r * exp((a * n * vt + r * exp(a) + vd) / (n * vt)) / (n * vt)

	def grad_diode_current(self, vd, _wx=None):
		"""
		Gradient of the diode current at provided diode voltage
		:param vd: diode voltage [V], array of length N
		:param _wx: W(x), reduces number of calculations if provided
		:return: diode current [A], matrix of shape [N, 3] with N the number
			of voltage samples and 3 is the number of diode parameters
		"""
		a, n, r, vt = self.a, self.n, self.r, self.vt
		wx = w(self._x(vd)) if _wx is None else _wx
		return wx / (1 + wx) * stack((
			n * vt / r - exp(a) / wx,
			vt * wx / r - exp(a) / n - vd / (n * r),
			exp(a) / r - n * vt * wx / r ** 2
		))
