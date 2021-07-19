class Adam:
	"""
	ADAptive Moment estimation (ADAM) optimizer [1]

	[1]	Kingma, Diederik; Ba, Jimmy (2014). "Adam: A Method for Stochastic
		Optimization". arXiv:1412.6980
	"""

	def __init__(self, max_iter=1e5, lr=1e-3, b1=0.99, b2=0.999, eps=1e-8):
		"""
		:param max_iter: maximum number of iterations the optimization
			process is allowed to perform
		:param lr: learning rate, lower this if the optimization process is
			instabel, increase this if the optimization process is very slow.
		:param b1: beta1, forgetting factor of the first moment
		:param b2:  beta2, forgetting factor of the second moment
		:param eps: epsilon, prevents a division by zero, therefore
			stabilizing the optimization process
		"""
		self.max_iter = int(max_iter)
		self._lr = lr
		self._b1 = b1
		self._b2 = b2
		self._eps = eps
		self._t = 1
		self._m = 0
		self._v = 0

	def step(self, grad_y_pred):
		"""
		:param grad_y_pred: gradient of predicted output, matrix of size
			[N x P] with N the number of predictions and P the number of
			optimization parameters
		:return: optimization parameter step, array with length equal to
			number of optimization parameters
		"""
		m, v, t, lr, b1, b2, eps = \
			self._m, self._v, self._t, self._lr, self._b1, self._b2, self._eps
		m = b1 * m + (1 - b1) * grad_y_pred
		v = b2 * v + (1 - b2) * grad_y_pred ** 2
		m_ = m / (1 - b1 ** t)
		v_ = v / (1 - b2 ** t)
		self._m, self._v, self._t = m, v, t + 1
		return - lr * m_ / (v_ ** .5 + eps)
