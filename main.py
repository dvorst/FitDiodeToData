import numpy as np
import matplotlib.pyplot as plt
import lib
from pathlib import Path
import pandas


def main():
	# ShockleyResistanceDiodeModel.derive_equations()
	# exit()

	path = 'example_data.csv'

	if not Path('example_data.csv').exists():
		with open(path, 'w') as file:
			file.write('Vd [V];Id [mA]\n')
		print('example_data.csv file did not exist, created an empty one')
		exit()

	# read data into dataframe
	data = pandas.read_csv(path, sep=';', header=0)

	# check that its not empty
	if data.empty:
		print('empty data file')
		exit()

	# convert data to numpy arrays
	vds = data['Vd [V]'].values
	ids = data['Id [mA]'].values * 1e-3

	# fit diode model to it
	diode = lib.diode_models.ShockleyResistanceDiodeModel()
	diode.fit_model_to_data(ids, vds)

	# print model parameters
	print('-' * 80)
	print('i_s = %.3e' % diode.i_s)
	print('n = %.3f' % diode.n)
	print('R = %.3f' % diode.r)

	# plot data
	vde = np.linspace(vds[0], vds[-1], 100)
	ide = diode.diode_current(vde)
	plt.semilogy(vde, ide, label='fitted model')
	plt.semilogy(vds, ids, '*', label='data points')
	plt.ylabel('Vd [V]')
	plt.xlabel('Id [mA]')
	plt.title('Shockley Diode model with Series resistance')
	plt.show()


main()
