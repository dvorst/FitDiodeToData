from numpy import log as ln
from numpy import sum


class MSLE:
	"""
	Altered version of Mean Squared Error (MSE), it uses the natural
	logarithm in the calculation of the error, hence the name MSLE

		error = ln(y_pred) - ln(y_true)
	"""

	def __init__(self, y_true, trh):
		"""
		:param y_true: the true output values of the model
		:param trh: threshold, if the calculated loss reaches a value below
			this, the optimization process should stop
		"""
		self.thr = trh
		self._y_true = y_true
		self._N = len(y_true)

	def calculate(self, y_pred):
		"""
		:param y_pred: predicted output of the model
		:return: calculated MSLE
		"""
		e = self.error(y_pred)
		return sum(e) ** 2 / self._N

	def error(self, y_pred):
		"""
		:param y_pred: predicted output of the model
		:return: returns error, error = ln(y_pred) - ln(y_true)
		"""
		return ln(y_pred) - ln(self._y_true)

	def grad_loss(self, y_pred, grad_y_pred):
		"""
		:param y_pred: predicted output of the model
		:param grad_y_pred: gradient of the predicted output of the model
		:return: gradient of the loss
		"""
		e, de = self.error(y_pred), self.grad_error(y_pred, grad_y_pred)
		return sum(e * de, axis=1) * 2 / self._N

	@staticmethod
	def grad_error(y_pred, grad_y_pred):
		"""
		:param y_pred: predicted output of the model
		:param grad_y_pred: gradient of the predicted output of the model
		:return: gradient of the error
		"""
		return grad_y_pred / y_pred
