# Processing script for EBAI
import numpy as np
import sys
import os
import re
import time
import matplotlib
matplotlib.use('Agg') # for remote use on server
import matplotlib.pyplot as pl
from shlex import split as splitsh
import signal
import subprocess

# Greek lettering labels
lab1 = [r'$T_2/T_1$', r'$\rho_1 + \rho_2$', r'$e sin(\omega)$', r'$e cos(\omega)$', r'$sin(i)$']
lab2 = [r'$\alpha = T_2 / T_1$', r'$\beta = \rho_1 + \rho_2$', r'$\gamma = e sin(\omega)$', r'$\delta = e cos(\omega)$', r'$\epsilon = sin(i)$']
lab3 = [r'$e$', r'$\omega$' + ' [deg]', r'$\rho_2 / \rho_1$', r'$T_1$' + ' [K]', r'$T_2$' + ' [K]', r'$i$' + ' [deg]']
lab4 = [r'$T_2/T_1$', r'$M_2/M_1$', r'$(\Omega^I-\Omega)/(\Omega^I-\Omega^O)$', r'$sin(i)$', r'$potential$']
#Alpha = temp ratio, Beta = mass ratio, Gamma = fillout factor, Delta = sin(i)

# Create default settings file
def make_settings(values = ['501', 'dlc', 'lcs/', 'lcs/synthetic/', 'lcs/real/', 'jobs/', 'plots/', 'phoebe/', 'ann/', 'results/', ' ']):
	file = open('ebai.settings', 'w')
	
	file.write('%s\t= Data points per input lightcurve\n' %values[0])
	file.write('%s\t= Type of EB lightcurve (dlc - detached, olc - overcontact)\n' %values[1])
	file.write('%s\t= EB lightcurve directory\n' %values[2])
	file.write('%s\t= Generated lightcurve sub-directory\n' %values[3])
	file.write('%s\t= Real lightcurve sub-directory\n' %values[4])
	file.write('%s\t= Directory to save slurm job files to\n' %values[5])
	file.write('%s\t= Directory to save plots to\n' %values[6])
	file.write('%s\t= Directory with phoebe defaults file\n' %values[7])
	file.write('%s\t= Directory to save ANN output\n' %values[8])
	file.write('%s\t= Directory to save final parameter results\n' %values[9])
	file.write('%s\t= Email for slurm jobs\n' %values[10])
	file.close()
	
	return np.genfromtxt('ebai.settings', delimiter='\t', dtype=str).T[0], np.genfromtxt('ebai.settings', delimiter='\t', dtype=str).T[-1]
	
# Start-up instructions
def prompt():

	if 'subprocess' not in sys.argv: 
		print '\n\n\033[1;31m [EBAI] - Eclipsing Binaries via Artificial Intelligence\033[1;m\n'

		print '\033[1;34m  List of System Arguments\033[1;m\n'
		print '\033[1;34m  (0)\033[1;m\033[1;38m Edit ebai.settings file\033[1;m'
		print '\033[1;30m\t- File is automatically created if not in script directory'
		print '\033[1;30m\t- Script creates directories specified in ebai.settings'
		print '\033[1;34m  (1)\033[1;m\033[1;38m Compute synthetic LCs\033[1;m'
		print '\033[1;30m\t- Edit ebai.settings to make detached or overcontact EBs'
		print '\033[1;34m  (2)\033[1;m\033[1;38m Plot parameter distributions of synthetic LCs\033[1;m'
		print '\033[1;34m  (3)\033[1;m\033[1;38m Polyfit synthetic LCs\033[1;m'
		print '\033[1;34m  (4)\033[1;m\033[1;38m Reformat polyfitted synthetic LCs for ANN\033[1;m'
		print '\033[1;34m  (5)\033[1;m\033[1;38m Reformat polyfitted real LCs for ANN\033[1;m'
		print '\033[1;34m  (6)\033[1;m\033[1;38m Identify and separate detached and overcontact EBs from real distribution\033[1;m'
		print '\033[1;34m  (7)\033[1;m\033[1;38m Chi-squared based reduction of synthetic LC distributions\033[1;m'
		print '\033[1;34m  (8)\033[1;m\033[1;38m Train ANN with varying LRP & topologies\033[1;m'
		print '\033[1;34m  (9)\033[1;m\033[1;38m Plot ANN training results from LRP & topology variations\033[1;m'
		print '\033[1;30m\t- Idenifies optimal ANN parameters for training'
		print '\033[1;34m  (10)\033[1;m\033[1;38m Initialize training of ANN\033[1;m'
		print '\033[1;34m  (11)\033[1;m\033[1;38m Continue training of ANN\033[1;m'
		print '\033[1;34m  (12)\033[1;m\033[1;38m Run ANN recognition and plot results\033[1;m'
		print '\033[1;30m\t- Plots principal and physical parameter distributions\033[1;m\n'
	
		if len(sys.argv) == 1: # End if user provided no system argument
			exit() 
		if sys.argv[1] not in map(str, range(0, 14)): # Error message
			print '\n\033[1;31m  ERROR:\033[1;m\033[1;38m Invalid user input\n\n'
			exit()
		
		print '\033[1;34m  Mode:\t\033[1;m\033[1;38m' + sys.argv[1]

	return int(sys.argv[1]) # Return selection

mode = prompt()

# Check for settings file to load
try: 
	var = np.genfromtxt('ebai.settings', delimiter='\t', dtype=str).T[0]
	des = np.genfromtxt('ebai.settings', delimiter='\t', dtype=str).T[-1]

# Create default settings file if none found
except: var, des = make_settings()

# Progess bar function
def progress(done, total):
	barLen, progress = 30, ''
	for i in range(barLen):
		if (i < int(barLen * done / total)): progress += '>'
		else: progress += ' '
	sys.stdout.write('\r')
	sys.stdout.write('\033[1;32m    [%s]\033[1;m' %progress)
	sys.stdout.write('\033[1;38m - \033[1;m')
	sys.stdout.write('\033[1;31m%s\033[1;m' %str(done))
	sys.stdout.write('\033[1;38m of \033[1;m')
	sys.stdout.write('\033[1;31m%s\033[1;m' %str(total))
	sys.stdout.write('\033[1;38m processed (\033[1;m')
	sys.stdout.write('\033[1;31m%.2f%%\033[1;m' %(done * 100. / total))
	sys.stdout.write('\033[1;38m)\033[1;m')
	sys.stdout.flush()

# Create sbatch job
def make_job(command, job_name, output = 0, processors = 1, opdir = os.getcwd()+'/'):

	job = open(var[5] + job_name + '.sh', 'w')
	job.write('#!/bin/bash\n')
	job.write('\n#SBATCH -J %s' %job_name)
	job.write('\n#SBATCH -p big')
	job.write('\n#SBATCH -N 1')
	job.write('\n#SBATCH -n %i' %processors)
	job.write('\n#SBATCH -t 2-00:00:00')
	job.write('\n#SBATCH -o %s.out' %(var[5] + job_name))
	job.write('\n#SBATCH -D %s\n\n' %opdir)
	if '@' in var[10]:
		job.write('#SBATCH --mail-type=BEGIN,END,FAIL')
		job.write('\n#SBATCH --mail-user=%s\n\n' %var[10])
	
	# For list of commands
	if isinstance(command, list) == True:
		for x in range(0, len(command)):
			job.write(command[x])
			if output[x] != 0: job.write(' > '+ output[x] + '\n')
			else: job.write('\n')
	
	# For singular command
	else: 
		job.write(command)
		if output != 0: job.write(' > '+ output)
	
	job.close() # Close file
	
# Compute and save synthetic LCs
def compute_lcs(end, start = 0):

	# Get random principal parameters
	def random_params():
		if sys.argv[4] == 'dlc':

			alpha = 1-abs(0.18*np.sqrt(-2*np.log(rng(0,1)))*np.cos(2*np.pi*rng(0,1))) # Alpha = T2/T1: roughly the surface brightness ratio
			# Without any loss of generality we may assume that the primary eclipse is deeper, thus the primary star's surface 
			# brightness is larger. The value of alpha should thus be sampled on the [0,1] interval. To avoid problems with too 
			# small ratios, we actually sample on the[0.2,1] interval.

			beta = 0.05 + rng(0,0.45) # Beta = rho1 + rho2: Fractional radii sum of both stars
			# Geometrically it corresponds to the sine of the half-eclipse phase duration. Small values of beta correspond to well
			# detached binaries, whereas values aproaching unity correspond to semi-detached and sequentially overcontact systems. 
			# Both extremes will deteriorate the model stability, so we choose the value from the [0.05, 0.5] interval.

			e0max = 0.5*np.exp(-6*(beta-0.05))		# Attenuation factor (dependence of eccentricity on the sum of radii):
			ecc = e0max * -1/3 * np.log(rng(0,1))	# Eccentricity
			omega = rng(0, 2*np.pi)					# Argument of periastron
			gamma = ecc * np.sin(omega)
			delta = ecc * np.cos(omega)
			# Gamma & Delta: these two derived parameters are introduced to orthogonalize eccentricity and the argument of periastron. 
			# For example, when omega = pi/2 (which means that we look at the binary along the semi-major axis), this parameter will 
			# be equal unity and the next parameter will be equal zero. However, in order to get reasonable distribution over parameters, 
			# we cannot assume uniformity in gamma or delta; instead, we adopt exponential distribution in e, uniform distribution in 
			# omega, and then compute the values of gamma and delta.

			i_eclipse = np.arcsin(np.sqrt(1-(0.9*beta)**2))
			incl = i_eclipse + rng(0,(np.pi/2)-i_eclipse)
			epsilon = np.sin(incl)
			# Epsilon: this parameter is better than the inclination itself because it closely corresponds to the parameter beta - it is 
			# the sine of the angle that influences the duration and shape of the light curve. Since it doesn't make a whole lot of sense 
			# to sample detached binaries that don't exhibit eclipses, we choose a value from the [i_eclipse,1] interval. The i_eclipse
			# value depends on beta and the eccentricity, where we actually use an upper limit for grazing eclipses (omega = +/-pi/2)

			# IMPORTANT: Because of the numeric instability of Omega(q) for small q,
			# the roles of stars are changed here: q > 1, star 1 is smaller and cooler,
			# star 2 is hotter and larger.

			# Return the acquired random set
			return [alpha, beta, gamma, delta, epsilon, ecc, omega, incl]

		if sys.argv[4] == 'olc':

			# Alpha = temperature ratio:
			alpha = 1/(1-abs(0.14*np.sqrt(-2*np.log(rng(0,1)))*np.cos(2*np.pi*rng(0,1)))) 

			# Beta = mass ratio:
			beta = 1/(1-0.22*abs(np.sqrt(-2*np.log(rng(0,1)))*np.cos(2*np.pi*rng(0,1))))

			# Potentials
			potL = critical_pot(beta, 1.0, 0.0)
			pot = potL[1] + rng(0, potL[0]-potL[1])
			
			# Gamma = fillout factor = (Omega(L1)-Omega)/(Omega(L1)-Omega(L2))
			gamma = (potL[0]-pot)/(potL[0]-potL[1])

			# Delta = sin(i)
			delta = 0.2 + rng(0,0.8)
			ecc = 0.0
			omega = np.pi/2

			# Return the acquired random set:
			return [alpha, beta, gamma, delta, pot, ecc, omega]

	# Calculate SBR parameters randomly
	def sbr_params(pars):
		if sys.argv[4] == 'dlc':
			# This function takes the five canonical parameters contained in the passed
			# array pars (labelled alpha thru epsilon above) and computes surface
			# brightness parameters, namely radii and temperatures. A combination
			# of these 4 SBR parameters that yields the right values of passed
			# canonical parameters is a tie.

			# rho1:
			# this parameter is obtained by sampling from the [0.025,beta-0.025]
			# interval. The interval assures that the size of either star is not
			# smaller than 0.025 of the semimajor axis.

			r2r1 = 1+0.25*np.sqrt(-2*np.log(rng(0,1)))*np.cos(2*np.pi*rng(0,1))
			rho1 = pars[1]/(1+r2r1)
			rho2 = pars[1]*r2r1/(1+r2r1)

			#set rho1 = 0.025 + rand(0, pars[2]-0.05)
			# rho2:
			# this parameter is obtained by subtracting rho1 from beta:
			#set rho2 = pars[2] - rho1

			# T1:
			# the selection for T1 would most easily be  done by simply assuming some
			# fixed value for T1, i.e. 6000K or such. However, since stars are not
			# perfect black bodies, parameter alpha is not really an ideal measure of
			# the SBR. We will thus resort to random sampling one more time, choosing
			# a surface temperature on the [5000K, 30000K] interval. This way gravity
			# darkening, limb darkening, reflection and other secular effects will
			# introduce a systematic scatter that will actually be used to assess the
			# expected 'bin' width of this crude 5-parameter model. The 'bin' in this
			# context corresponds to the span of actual physical parameters that yield
			# the same values of canonical parameters alpha thru epsilon.

			T2 = 0
			while(T2 < 3500):
				T1 = 5000 + rng(0,25000)
				T2 = T1 * pars[0]

			return rho1, rho2, T1, T2 # Return

		if sys.argv[4] == 'olc':
			T1 = 0
			while(T1 < 3500):
				T2 = 3500 + rng(0,3500)
				T1 = T2/pars[0]

			return pb.getpar('phoebe_radius1')/10., pb.getpar('phoebe_radius2')/10., T1, T2 # Return
			#return -1, -1, T1, T2 # Return

	# Calculate effective potential (Omega_1) of the primary star
	def primary_pot(D, q, r, F, lmbda, nu):
		
		# Arguments:
		#   D          ..  instantaneous separation between components in units of semi-major axis (a)
		#   q          ..  mass ratio (secondary over primary)
		#   r          ..  star radius in units of semi-major axis (a)
		#   F          ..  synchronicity parameter
		#   lambda     ..  direction cosine
		#   nu         ..  direction cosine
	
		return 1/r + q*(D**2 + r**2 -2*r*lmbda*D)**(-1/2) - r*lmbda/D**2 + (1/2)*(F**2)*(1+q)*(r**2)*(1-nu**2)

	# Calculate effective potential (Omega_2) of the secondary star
	def secondary_pot(D, q, r, F, lmbda, nu):
		
		# Arguments:
		#   D          ..  instantaneous separation between components in units of semi-major axis (a)
		#   q          ..  mass ratio (secondary over primary)
		#   r          ..  star radius in units of semi-major axis (a)
		#   F          ..  synchronicity parameter
		#   lambda     ..  direction cosine
		#   nu         ..  direction cosine
	
		q = 1/q
		pot1 = primary_pot(D, q, r, F, lmbda, nu)
		return pot1/q + (1/2)*(q-1)/q # Return

	# Calculate critical effective potentials of both stars
	def critical_pot(q, F, e):

		# These are the potentials that go through L1 (Omega{crit}^1) and L2
		# (Omega{crit}^2).

		D, xL, dxL = 1-e, 0.5, 1.1e-6
	
		while(abs(dxL) > 1e-6):
			xL = xL + dxL
			Force = F**2*(q+1)*xL - 1/xL**2 - q*(xL-D)/abs((D-xL)**3) - q/D**2
			dxLdF = 1/(F**2*(q+1) + 2/xL**3 + 2*q/abs((D-xL)**3))
			dxL = -1 * Force * dxLdF
		
		L1crit = 1/xL + q*(D**2 + xL**2 -2*xL*D)**(-1/2) - xL/D**2 + 1/2*(F**2)*(1+q)*(xL**2)

		if(q > 1): q2 = 1/q 
		else: q2 = q

		D, F, dxL = 1, 1, 1.1e-6
		factor = (q2/3/(q2+1))**(1/3)
		xL = 1 + factor + 1/3*factor**2 + 1/9*factor**3
	
		while(abs(dxL) > 1e-6):
			xL = xL + dxL
			Force = F**2*(q2+1)*xL - 1/xL**2 - q2*(xL-D)/abs((D-xL)**3) - q2/D**2
			dxLdF = 1/(F**2*(q2+1) + 2/xL**3 + 2*q2/abs((D-xL)**3))
			dxL = -1 * Force * dxLdF

		if(q > 1): xL = D - xL
		L2crit = 1/abs(xL) + q*(1/abs(xL-1)-xL) + 1/2*(q+1)*xL**2

		return L1crit, L2crit # Return

	# Calculate phase of conjunction
	def conjunction_phase(ecc, omega):
		
		ups_c = np.pi/2 - omega
		E_c = 2 * np.arctan(np.sqrt((1-ecc)/(1+ecc)) * np.tan(ups_c/2))
		M_c = E_c - ecc * np.sin(E_c)

		return (M_c + omega)/2/np.pi - 0.25

	# Set default model parameters
	def default_params(pars, sbrs):

		if var[1] == 'dlc':
			pb.setpar('phoebe_ecc', pars[5])
			pb.setpar('phoebe_perr0', pars[6])
			pb.setpar('phoebe_incl', 180*np.arcsin(pars[4])/np.pi)
			pb.setpar('phoebe_pot1', primary_pot(1-pars[5], 1, sbrs[0], 1, 1, 0))
			pb.setpar('phoebe_pot2', secondary_pot(1-pars[5], 1, sbrs[1], 1, 1, 0))
			pb.setpar('phoebe_teff1', sbrs[2])
			pb.setpar('phoebe_teff2', sbrs[3])
			pb.setpar('phoebe_pshift', -1*conjunction_phase(pars[5], pars[6]))
			#pb.setpar('phoebe_lc_filter', 'Kepler:mean', 0)

		if var[1] == 'olc':
			pb.setpar('phoebe_incl', 180*np.arcsin(pars[3])/np.pi)
			pb.setpar('phoebe_rm', pars[1])
			pb.setpar('phoebe_pot1', pars[4])
			pb.setpar('phoebe_pot2', pars[4])
			pb.setpar('phoebe_teff1', sbrs[2])
			pb.setpar('phoebe_teff2', sbrs[3])
			pb.setpar('phoebe_pshift', 0.5)
			pb.setpar('phoebe_lc_filter', 'Kepler:mean', 0)

	# Set value of gravity darkening
	def grav_darkening(sbrs):

		# Takes SBR parameters and, based on the temperature, sets the value 
		# of gravity darkening exponent. The regime changes discretely at T=7500K.

		if sys.argv[4] == 'dlc': lim = 7500
		if sys.argv[4] == 'olc': lim = 7000

		if (sbrs[2] < lim):
			pb.setpar('phoebe_grb1', 0.32)		# 0.32 for convective envelopes
		else: pb.setpar('phoebe_grb1', 1.00)	# 1.0 for radiative envelopes

		if (sbrs[3] < lim):
			pb.setpar('phoebe_grb2', 0.32)		# 0.32 for convective envelopes
		else: pb.setpar('phoebe_grb2', 1.00)	# 1.0 for radiative envelopes

	# Verify parameters feasibility
	def feasible(pars, sbrs):

		if sys.argv[4] == 'dlc':
			T2T1 = pars[0]	# -1 test: is T2/T1 less than 0.2?
			ecc = pars[5]	# Zero test: is eccentricity more than 0.8?

			# Test 1: the sum of radii comparable to the distance between both stars close to periastron?
			summ = 1.5 * pars[1] # 1.5 is arbitrary, pars[1] is the sum of radii

			# Test 2: are periastron potentials overflowing the lobe?
			critpot = critical_pot(1, 1, ecc)
			pot1 = primary_pot(1-pars[5], 1, sbrs[0], 1, 1, 0)
			pot2 = secondary_pot(1-pars[5], 1, sbrs[1], 1, 1, 0)

			# Determine feasibility
			if(pars[0] < 0.2 or ecc > 0.8 or summ > 1-ecc or pot1 < critpot[0] or pot2 < critpot[0]): 
				state = 'false'
			else: state = 'true'
			
			return state # Return

		if sys.argv[4] == 'olc':
			state = 'true'

			if (pars[1] < 0.15 or pars[1] > 1/0.15):
				state = 'false'
			if (np.arcsin(pars[3]) < np.arccos(sbrs[0]+sbrs[1])):
				state = 'false'

			return state

	# Function imports
	import phoebeBackend as pb
	from numpy.random import uniform as rng

	pb.init(); pb.configure()	# Startup phoebe
	if sys.argv[4] == 'dlc': 	# Open a generic dlc  model
		pb.open(var[7] + 'detached.phoebe')
	if sys.argv[4] == 'olc': 	# Open a generic olc model
		pb.open(var[7] + 'overcontact.phoebe')
	accept = 0 					# Counter for accepted LCs
    
	# Phase tuple on [-0.5, 0.5] interval with 'points' vertices
	phases = tuple(np.linspace(-0.5, 0.5, int(var[0])).tolist())

	for i in range((end-start)*50):			# LC generation loop
		pars = random_params()				# Principal parameters
		sbrs = sbr_params(pars)				# SBR parameters
		if(feasible(pars, sbrs) == 'true'):	# Check parameter feasibility
			default_params(pars, sbrs) 		# Load default parameters
			pb.updateLD()					# Limb darkening [coefficients from Van Hamme (1993)]
			grav_darkening(sbrs)			# Gravity darkening
			flux = pb.lc(phases, 0)			# Get flux
			
			# Time out, NaN, and 0 testing
			if flux != False and True not in np.isnan(flux) and flux[0] != 0:
				if sys.argv[4] == 'dlc': 
					# Check if the luminosities are not outside the expected interval:
					sbr1 = pb.getpar('phoebe_sbr1')
					sbr2 = pb.getpar('phoebe_sbr2')		
					if(sbr2/sbr1 >= 0.1 and sbr2 <= sbr1):	# Accept curve, create file
									
						# LC file name
						name = var[3] + str(accept + start) + '.' + sys.argv[4]
						fileout = open(name, 'w')	# Create LC file

						# Create file header
						fileout.write('# alpha   = %s\n' %str(pars[0]))
						fileout.write('# beta    = %s\n' %str(pars[1]))
						fileout.write('# gamma   = %s\n' %str(pars[2]))
						fileout.write('# delta   = %s\n' %str(pars[3]))
						fileout.write('# epsilon = %s\n' %str(pars[4]))
						fileout.write('# ecc     = %s\n' %str(pars[5]))
						fileout.write('# omega   = %s\n' %str(pars[6]))
						fileout.write('# rho1    = %s\n' %str(sbrs[0]))
						fileout.write('# rho2    = %s\n' %str(sbrs[1]))
						fileout.write('# Teff1   = %s\n' %str(sbrs[2]))
						fileout.write('# Teff2   = %s\n' %str(sbrs[3]))
						fileout.write('# SBR1    = %s\n' %str(sbr1))
						fileout.write('# SBR2    = %s\n' %str(sbr2))

						# Compile data points array
						data = [4*np.pi/(pb.getpar('phoebe_plum1')+pb.getpar('phoebe_plum2')) * flux[j] for j in range(int(var[0]))]

						# Write phases and fluxes to file
						for j in range(int(var[0])):
							fileout.write('%s\t%s\n' %(phases[j], str(data[j])))

						accept += 1						# Kick up counter of accepted LCs
						if accept+start == end: break	# End program for 'end' accepted LCs 
						fileout.close() 				# Close output file

				if sys.argv[4] == 'olc':

					# LC file name
					name = var[3] + str(accept + start) + '.' + sys.argv[4]
					fileout = open(name, 'w')	# Create LC file
										
					# Create file header
					fileout.write('# alpha   = %s\n' %str(pars[0]))
					fileout.write('# beta    = %s\n' %str(pars[1]))
					fileout.write('# gamma   = %s\n' %str(pars[2]))
					fileout.write('# delta   = %s\n' %str(pars[3]))
					fileout.write('# pot     = %s\n' %str(pars[4]))
					fileout.write('# ecc     = %s\n' %str(pars[5]))
					fileout.write('# omega   = %s\n' %str(pars[6]))
					fileout.write('# rho1    = %s\n' %str(sbrs[0]))
					fileout.write('# rho2    = %s\n' %str(sbrs[1]))
					fileout.write('# Teff1   = %s\n' %str(sbrs[2]))
					fileout.write('# Teff2   = %s\n' %str(sbrs[3]))
					fileout.write('# SBR1    = %s\n' %str(0.0))
					fileout.write('# SBR2    = %s\n' %str(0.0))

					# Compile data points array
					data = [4*np.pi/(pb.getpar('phoebe_plum1')+pb.getpar('phoebe_plum2')) * flux[j] for j in range(0, int(var[0]))]

					# Write phases and fluxes to file
					for j in range(int(var[0])):
						fileout.write('%s\t%s\n' %(phases[j], str(data[j])))

					accept += 1						# Kick up counter of accepted LCs
					if accept+start == end: break	# End program for 'end' accepted LCs 
					fileout.close() 				# Close output file
	
	pb.quit() # Quit phoebe

# Plot parameter distributions
def plot_params(list_file, bounds_save):

	# Create boundary file	
	def save_bounds(pars):
		file = open(bounds_save, 'w')
		file.write('%s\t%s\n' %(str(min(pars[0])), str(max(pars[0]))))
		file.write('%s\t%s\n' %(str(min(pars[1])), str(max(pars[1]))))
		file.write('%s\t%s\n' %(str(min(pars[2])), str(max(pars[2]))))
		file.write('%s\t%s\n' %(str(min(pars[3])), str(max(pars[3]))))
		if var[1] == 'dlc': file.write('%s\t%s\n' %(str(min(pars[4])), str(max(pars[4]))))

	# Lightcurve file names
	files = np.loadtxt(list_file, dtype=str)

	pars = load_params(files) # Extract principal parameters

	# Plot parameter distributions
	print '\n\n Plotting parameter distributions...\n'
	if var[1] == 'dlc': lab = lab1
	if var[1] == 'olc': lab = lab4	
	tag = var[1] + '.' + str(len(files) / 1000) + 'k' # Plot tag
	plot_dist(pars[0], lab[0], 'alpha.gen.%s.png' %tag)
	plot_dist(pars[1], lab[1], 'beta.gen.%s.png' %tag)
	plot_dist(pars[2], lab[2], 'gamma.gen.%s.png' %tag)
	plot_dist(pars[3], lab[3], 'delta.gen.%s.png' %tag)
	if var[1] == 'dlc': plot_dist(pars[4], lab[4], 'epsilon.gen.%s.png' %tag)
	if var[1] == 'olc': plot_dist(pars[4], lab[4], 'potential.gen.%s.png' %tag)
	plot_dist(pars[5], lab3[0], 'eccentricity.gen.%s.png' %tag)
	plot_dist(pars[6], lab3[1], 'argument.gen.%s.png' %tag)
	if var[1] == 'dlc': plot_dist(pars[8]/pars[7], lab3[2], 'radii.gen.%s.png' %tag)
	plot_dist(pars[9], lab3[3], 'temp.gen.%s.png' %tag)
	plot_dist(pars[10], lab3[4], 'temp.gen.%s.png' %tag)
	plot_dist(pars[11], lab3[5], 'inclination.gen.%s.png' %tag)
	
	save_bounds(pars) # Save boundary file

# Load paramters from non-polyfitted synthetic LC files
def load_params(files):
	print '\n\n  Loading parameters from lightcurve files...\n'

	# Initialize lists
	alpha, beta, gamma, delta, epsilon = [], [], [], [], []
	temp1, temp2, rho1, rho2, omega, ecc = [], [], [], [], [], []
	
	progress(0, len(files)) # Start progress

	for i in range(0, len(files), 1):
		lc = open(files[i], 'r') # Open file

		# Get principal parameters
		alpha.append(float(lc.readline()[12:]))
		beta.append(float(lc.readline()[12:]))
		gamma.append(float(lc.readline()[12:]))
		delta.append(float(lc.readline()[12:]))
		epsilon.append(float(lc.readline()[12:])) # Pot for olc, epilson for dlc

		# Get physical parameters
		ecc.append(float(lc.readline()[12:]))
		omega.append(float(lc.readline()[12:]))
		rho1.append(float(lc.readline()[12:]))
		rho2.append(float(lc.readline()[12:]))
		temp1.append(float(lc.readline()[12:]))
		temp2.append(float(lc.readline()[12:]))

		progress(i+1, len(files)) # Update progress  
	
	# Bundle parameters
	a = [np.array(alpha), np.array(beta), np.array(gamma), np.array(delta), np.array(epsilon)]
	if var[1] == 'dlc': incl = 180*np.arcsin(a[4])/np.pi
	if var[1] == 'olc': incl = 180*np.arcsin(a[3])/np.pi
	b = [np.array(ecc), np.array(omega)*180/np.pi, np.array(rho1), np.array(rho2), np.array(temp1), np.array(temp2), incl]	
	return a[0], a[1], a[2], a[3], a[4], b[0], b[1], b[2], b[3], b[4], b[5], b[6]

# Plot distribution
def plot_dist(param_vals, label, save_file):
	weight = np.ones_like(param_vals)/len(param_vals)
	f = pl.figure(save_file, figsize = (6, 4))
	h = pl.hist(param_vals, bins=50, weights=weight*100, histtype='bar', align='mid', orientation='vertical', label=label)
	pl.legend(loc='best')
	pl.xlabel('Value'); pl.ylabel('Frequency (%)')
	pl.xlim([min(param_vals), max(param_vals)])
	pl.savefig(var[6] + 'parameters/' + save_file)
	pl.close(f)

# Polyfit LCs	
def polyfit_lcs(list_file, njobs, pf_order = 2, iters = 10000, phase_col = 0, flux_col = 1):
	print '\n  Polyfitting LCs...\n'

	# Lightcurve file names
	files = np.split(np.loadtxt(list_file, dtype=str), njobs)[int(sys.argv[4])-1]
	
	# Polyfit command - see 'polyfit' for argument info
	command = 'polyfit -o %i -i %i -n %s -c %i %i --find-knots --find-step ' %(pf_order, iters, var[0], phase_col, flux_col)
	
	# Name of files to be created
	output = [x.split('/')[-1].replace(var[1], 'pf.%s' %var[1]) for x in files]
	
	progress(0, len(files))	# Start progress
	for x in range(len(files)):
		os.system(command + files[x] + ' > ' + var[3] + output[x])
		progress(x+1, len(files)) # Update progress
		
# Create ouput text files
def make_file(arr, save_dir, prefix = 0, save_file = 'neverused', list_file = 'list.ann.%s.txt' %var[1]):
	if prefix == 0: prefix = save_dir

	if isinstance(save_file, str) == False:			# Check for list type
		progress(0, len(arr)) 						# Start progess
		list = open(save_dir + list_file, 'w') 		# Create list file
		for x in range(0, len(arr), 1):
			file = open(save_dir + save_file[x], 'w') 	# Create data file
			list.write('%s\n' %(prefix + save_file[x])) # Print list line
			for i in range(0, len(arr[x]), 1):		
				file.write('%s\n' %str(arr[x][i])) 	# Print data line
			progress(x+1, len(arr)) 				# Update progress
	
	else:										# For lone string, single file to save
		file = open(save_dir + save_file, 'w') 	# Create file
		for i in range(0, len(arr), 1):	
			file.write('%s\n' %str(prefix + arr[i])) 	# Print line
			#file.write('%s\n' %str(arr[i]))
	
# Reformat lightcurves for EBAI
def reformat_lcs():
	if sys.argv[3] == 'synthetic': # Reformat synthetic lightcurves
	
		# Get lightcurve file names
		lcs = np.loadtxt(var[3] + 'list.%s.txt' %var[1], dtype=str)
		pfs = np.loadtxt(var[3] + 'list.pf.%s.txt' %var[1], dtype=str)

		flux = load_lcs(pfs) 	# Read-in flux
		pars = load_params(lcs) # Extract principal parameters

		print '\n\n  Reformatting and saving synthetic LCs...\n'
		progress(0, len(flux)) 										# Start progress
		name_file = open(var[3] + 'list.par.pf.%s.txt' %var[1], 'w') 	# Create file
		
		for i in range(0, len(flux), 1):
		
			save_file = pfs[i].split('/')[-1].replace('pf', 'par.pf')
			#save_file = '%i.par.pf.%s' var[1]
			file = open(var[3] + save_file, 'w') # Create LC file
			name_file.write('%s\n' %(var[3] + save_file))    # Save name to list
			for k in range(0, len(flux[i]), 1): 		     # Save flux
				file.write('%s\n' %str(flux[i][k] / np.median(flux[i])))

			# Save principal parameters
			file.write('\n%s\n' %str(pars[0][i]))
			file.write('%s\n' %str(pars[1][i]))
			file.write('%s\n' %str(pars[2][i]))
			file.write('%s\n' %str(pars[3][i]))
			if var[1] == 'dlc': 
				file.write('%s' %str(pars[4][i]))

			progress(i+1, len(flux)) # Update progress 

	if sys.argv[3] == 'real': 	   # Reformat real lightcurves

		# Interpolate LC fluxes
		def interpolate_flux(id):

			fluxes, new_id = [], []	# Initialize lists
			progress(0, len(id))	# Start progress
			
			for i in range(0, len(id), 1):
				LC = np.loadtxt(var[4] + id[i], dtype=str)	# Extract data from file
				
				# Interpolate fluxes along x-axis
				phase = np.linspace(-0.5, 0.5, len(LC.T[len(LC[0])-1]))
				xvals = np.linspace(-0.5, 0.5, int(var[0])+1)
				interp = np.interp(xvals, phase, np.asarray(LC.T[len(LC[0])-1], dtype=np.float64))
				if np.median(interp) != 0:						# Throw out LC is median return failed
					new_id.append(id[i])					 	# Save name if accepted
					fluxes.append(interp / np.median(interp)) 	# Append to flux list
				
				progress(i+1, len(id))	# Update progress
			return fluxes, new_id		# Return lightcurve data

		# Get lightcurve file names
		keps = np.loadtxt(var[4] + 'list.data.txt', dtype=str)
		ids = np.loadtxt(var[4] + 'list.ebs.txt', dtype=str).T[0] # List generated by keplerebs.villanova.edu
		
		# Real: Find EBs in keps
		print '\n\n  Identifying Real EBs from KIC list...\n'; ebs = []
		progress(0, len(ids))							# Start progress
		for i in range(0, len(ids), 1): 
			match = [s for s in keps if ids[i] in s]	# Find match
			if len(match) >= 1: ebs.append(match[0])	# Record match
			progress(i+1, len(ids))						# Update progress

		# Read-in flux 
		print '\n\n  Reading in and interpolating real flux data...\n'
		klcs, ids = interpolate_flux(ebs)

		# Save reformatted curves
		print '\n\n  Saving reformatted real LCs...\n'
		make_file(ids, save_dir = var[4], save_file = 'list.ids.txt')
		make_file(klcs, save_dir = var[4], save_file = ['%i.pf.lc' %x for x in range(len(klcs))], list_file = 'list.pf.txt')

# Load LC files
def load_lcs(files):
	print '\n\n  Loading flux data...\n'

	flux_list = []			# Initialize list
	progress(0, len(files))	# Start progress
	
	for i in range(0, len(files), 1): # Extract data from file
		LC = np.loadtxt(files[i], dtype=str)
		
		# Save flux column
		if isinstance(LC.T[-1], np.ndarray) == False:
			flux = [LC.T[x] for x in range(0, len(LC.T))]
		else:
			flux = [LC.T[-1][x] for x in range(0, len(LC.T[-1]))]
			
		flux_list.append(flux)		# Store in flux list
		progress(i+1, len(files))	# Update progress   
	
	return np.array(flux_list, dtype=np.float64) # Return lightcurve data

# Chi-based reduction of LC distributions	
def reduce_lcs():

	# Compute chi values, similarity of d1 to d2 set
	def compute_chis(percents, d1, d2, d2names, tag):
		progress(0, len(d1))	 				# Start progress
		
		# Establish dictionaries
		step = percents[1] - percents[0]
		best = {}; rep = {}; unrep = {}
		
		# Add keys to dictionaries
		for x in percents:	
			best[x], rep[x], unrep[x] = [], [], []

		# Loop to compute chis
		for i in range(0, len(d1), 1):
			chis, names = [], []
			for x in range(0, len(d2), 1):			
				val = ((d1[i] - d2[x])**2. / 1.0).sum()	# Compute chi
				if val < 5:								# Append if not too great
					chis.append(val); names.append(d2names[x]) 

			#plot_dist(chis, str(i), str(i) + tag)
			#plot_dist(chis, '%i' %i, dist + '%i' %i, tag)	# Plot chi distribution

			# Sort arrays
			yx = zip(chis, names)
			yx.sort()
			sorted_names = [x for y, x in yx]
			chis.sort()

			# Get best matches, vary with top-take percentile
			for x in percents:
				n = int(x / 100. * len(chis))	# d2 LCs to take [35% * 50,000 = 17,501]
				lcs = sorted_names[:n]				# Best LCs by chis			
				for k in range(0, len(lcs), 1):		# Take best n LCs by chis
					if lcs[k] not in best[x]: best[x].append(lcs[k])
		
			progress(i+1, len(d1)) # Update progress
			
		# Label each d2 LC's relation to d1
		for x in percents:
			for a in range(0, len(d2names), 1):	
				if d2names[a] not in best[x]:
					unrep[x].append(d2names[a])	# LC not represented
				else: rep[x].append(d2names[a])	# Well represented LC
		
		# Percent accepted v. rejected LCs
		pl.figure(tag + '1', figsize = (8, 6))
		for x in percents:
			pl.plot(x, len(unrep[x]), 'ro')
		pl.legend(loc='best')
		pl.xlabel('Percentage of Best Fit LCs Accepted [%]')
		pl.ylabel('Rejected LCs (out of %s)' %str(len(d2names)))
		#pl.savefig(var[6] + 'rejected.%s.%s.png' %(var[1], tag))
		
		# Percent accepted v. accepted LCs
		pl.figure(tag + '2', figsize = (8, 6))
		for x in percents:
			pl.plot(x, len(unrep[x]), 'bo')
		pl.legend(loc='best')
		pl.xlabel('Percentage of Best Fit LCs Accepted [%]')
		pl.ylabel('Accepted LCs (out of %s)' %str(len(d2names)))
		#pl.savefig(var[6] + 'accepted.%s.%s.png' %(var[1], tag))
		
		return rep, unrep # Return
	
	# Only runs in 'dlc' mode when filtering detached from overcontact
	if sys.argv[3] == 'real' and var[1] == 'olc': var[1] = 'dlc'
	
	# Get file names
	gens = np.loadtxt(var[3] + 'list.par.pf.%s.txt' %var[1], dtype=str)
	keps = np.loadtxt(var[4] + 'list.pf.txt', dtype=str)
	kid = np.loadtxt(var[4] + 'list.ids.txt', dtype=str)

	# Get LC distributions
	glcs, klcs = load_lcs(gens), load_lcs(keps)

	# Get fluxes from generated LC array
	gflux = [glcs[x][:int(var[0])] for x in range(0, len(glcs))]
	klcs = [klcs[x][:int(var[0])] for x in range(0, len(klcs))]
	
	# Reinsert spacing between flux and parameters
	glcs = glcs.astype(str)
	glcs = [np.insert(glcs[x], int(var[0])+1, '') for x in range(0, len(glcs))]
	
	if sys.argv[3] == 'real': # Identify detached and overcontact LCs
	
		# Establish dictionaries
		reject = {'id': [], 'flux': []}
		accept = {'id': [], 'flux': []}

		trgt = 800 								# Estimated number of real EBs over 0.7 Morph
		percents = np.arange(1.0, 6.50, 0.5)	# Percentiles to check, starts at 0.5%
		
		# Synthetic: Look for similar real curves
		print '\n\n  Computing real LC differences relative to Synthetic LCs...\n'
		rep, unrep = compute_chis(percents, gflux, klcs, keps, 'gen-v-keps')

		# Find percentile giving len(unrep[x]) closest to 'trgt'
		print '\n\n  Indentifying best percentile acceptance and writing results to .txt files...\n'
		lens = [len(unrep[x]) for x in percents]
		x = percents[(np.abs([x - trgt for x in lens])).argmin()]	# Set dictionary index 

		for i in range(0, len(unrep[x])): 		# Find rejected
			if unrep[x][i] in keps: 			# Record rejected ID
				n = np.where(keps == unrep[x][i])[0][0]
				reject['flux'].append(klcs[n])	# Record rejected flux
				reject['id'].append(kid[n].split('/')[-1].split('.')[0])
		for i in range(0, len(rep[x])): 	 	# Find accepted
			if rep[x][i] in keps: 	 			# Record accepted ID
				n = np.where(keps == rep[x][i])[0][0]
				accept['flux'].append(klcs[n])	# Record accepted flux
				accept['id'].append(kid[n].split('/')[-1].split('.')[0])

		# Create output .txt files
		make_file(accept['id'], var[4], save_file = '%s%%.%i.ids.dlc.txt' %(str(x), len(accept['id'])))
		make_file(reject['id'], var[4], save_file = '%s%%.%i.ids.olc.txt' %(str(x), len(reject['id'])))

		# Save identified detached LCs
		print '\n  Saving accepted real LCs...\n'
		make_file(accept['flux'], save_dir = var[4], save_file = ['%i.ann.pf.dlc' %(x+1) for x in range(len(accept['flux']))], list_file = 'list.ann.pf.dlc.txt')

		# Save identified overcontact LCs
		print '\n\n  Saving rejected real LCs...\n'
		make_file(reject['flux'], save_dir = var[4], save_file = ['%i.ann.pf.olc' %(x+1) for x in range(len(reject['flux']))], list_file = 'list.ann.pf.olc.txt')

	if sys.argv[3] == 'synthetic': # Trim synthetic LC distribution
		
		# Get file names
		if var[1] == 'dlc': keps = np.loadtxt(var[4] + 'list.ann.pf.%s.txt' %var[1], dtype=str)
		if var[1] == 'olc': keps = np.loadtxt(var[4] + 'list.ann.pf.%s.txt' %var[1], dtype=str)
		
		#percents = np.arange(0.1, 1.0, 0.05)		# Percentiles to check, starts at 0.5%
		percents = np.arange(0.1, 0.16, 0.05)		# Percentiles to check, starts at 0.5%
		klcs = load_lcs(keps)					# Get LC distributions
		reject, accept = {'lc': []}, {'lc': []}	# Establish dictionaries
			
		# Synthetic: Look for similar real curves
		print '\n\n  Computing synthetic LC differences relative to real LCs...\n'
		rep, unrep = compute_chis(percents, klcs, gflux, gens, 'kep-v-gens')
		
		# Find percentile giving len(rep) closest to 7*len(gens)
		print '\n\n  Indentifying best percentile acceptance and writing results to .txt files...\n'
		lens = [len(rep[x]) for x in percents]
		x = percents[(np.abs([x - 7*len(klcs) for x in lens])).argmin()]	# Set dictionary index

		for i in range(0, len(unrep[x])): 	# Find rejected
			if unrep[x][i] in gens: 		# Record rejected flux
				reject['lc'].append(glcs[np.where(gens == unrep[x][i])[0][0]])	
		for i in range(0, len(rep[x])): 	# Find accepted
			if rep[x][i] in gens: 	 		# Record accepted flux
				accept['lc'].append(glcs[np.where(gens == rep[x][i])[0][0]])
		
		# Create output .txt files
		reject_file = 'rejected.%s%%.%i.gens.%s.txt' %(str(x), len(unrep[x]), var[1])
		accept_file = 'accepted.%s%%.%i.gens.%s.txt' %(str(x), len(rep[x]), var[1])
		make_file([a.split('/')[-1].replace('.par.pf', '') for a in unrep[x]], var[3], prefix = var[3], save_file = reject_file)
		make_file([a.split('/')[-1].replace('.par.pf', '') for a in rep[x]], var[3], prefix = var[3], save_file = accept_file)
	
		# Save synthetic LCs to genEBs/chi_lcs/
		print '\n  Saving accepted synthetic LCs...\n'
		make_file(accept['lc'], save_dir = var[3], save_file = ['%i.ann.par.pf.%s' %(x+1, var[1]) for x in range(len(accept['lc']))], list_file = 'list.ann.par.pf.%s.txt' %var[1])
		#print '\n\n  Saving rejected synthetic LCs...\n'
		#make_file(reject['lc'], save_dir = var[3], save_file = ['%i.ann.par.pf.%s' %(x+1, var[1]) for x in range(len(reject['lc']))], list_file = 'list.ann.par.pf.%s.txt' %var[1])
		
		#if var[1] == 'olc': accept_file = 'accepted.0.1%.5079.gens.olc.txt'
		#if var[1] == 'dlc': accept_file = 'accepted.0.1%.15828.gens.dlc.txt'

		# Plot parameter distributions
		plot_params(list_file = var[3]+accept_file, bounds_save = var[3]+'bounds.ann.par.pf.%s.txt' %var[1])

# Load ANN training output file
def load_train(files):

	cfs, val = [], []		# Initialize lists
	progress(0, len(files))	# Start progress
	
	for i in range(len(files)):						
		file = np.loadtxt(files[i], dtype=np.float64)	# Open file
		cfs.append(file.T[1])							# Add data to array
		progress(i+1, len(files))						# Update progress   

		# Identify and save parameter value
		value = files[i].split('/')[-1].split('.')[0]
		if files[i].split('/')[-1].split('.')[1].isdigit() == True:
			value = value + '.' + files[i].split('/')[-1].split('.')[1]
			val.append(float(value))
		else: 
			value = value.split('.')[0]
			val.append(int(value))
	
	# Sort arrays
	yx = zip(val, cfs)
	yx.sort()
	cfs = [x for y, x in yx]
	val.sort()

	return cfs, val	# Return
	
# Plot LRP
def plot_lrp(cfs, val, tag):
	
	# Plot LRP
	pl.figure(1, figsize = (8, 8))
	for x in range(0, len(cfs), 1):
		pl.plot(np.linspace(1, len(cfs[0]), len(cfs[0])), cfs[x], label=str(val[x]))
	pl.legend(loc='best')
	pl.xlabel('Training Iterations')
	pl.ylabel('Cost Function')
	#pl.ylim(0, 10000)
	pl.savefig(var[6] + 'lrp.%s.%s.png' %(var[1], tag))
	
	# Get last elements
	fcfs = [cfs[x][-1] for x in range(0, len(cfs))]
	
	# Display minimum
	print '  Fastest converging LRP = %s\n' %str(val[fcfs.index(min(fcfs))])
	
# Plot hidden layers
def plot_hidden(cfs, val, tag):

	width = 0.8 # Plot parameter
	
	# Get last elements
	fcfs = [cfs[x][-1] for x in range(0, len(cfs))]
		
	# Plot
	pl.figure(2, figsize = (8, 8))
	#pl.title('Hidden layer count effect on Cost Function')
	pl.bar(np.arange(len(cfs)), fcfs, width)
	pl.xticks(np.arange(len(cfs)) + width/2.0, val)
	pl.xlabel('Hidden Layers')
	pl.ylabel('Cost Function (%s training iterations)' %str(len(cfs[0])))
	pl.savefig(var[6] + 'hidden.%s.%s.png' %(var[1], tag))	

# Plot parameter mapping
def plot_mapping(data, name):
	print '\n  Plotting parameter mapping...'
	
	# Plot values
	pl.figure(name, figsize = (6, 8))
	s1 = pl.scatter(data.T[1][data.T[0] != 0.5], data.T[0][data.T[0] != 0.5], c='r', marker='o')
	s2 = pl.scatter(data.T[3][data.T[2] != 0.5], data.T[2][data.T[2] != 0.5]+0.5, c='b', marker='o')
	s3 = pl.scatter(data.T[5][data.T[4] != 0.5], data.T[4][data.T[4] != 0.5]+1.0, c='g', marker='o')
	s4 = pl.scatter(data.T[7][data.T[6] != 0.5], data.T[6][data.T[6] != 0.5]+1.5, c='y', marker='o')
	if var[1] == 'dlc': 
		s5 = pl.scatter(data.T[9][data.T[8] != 0.5], data.T[8][data.T[8] != 0.5]+2.0, c='c', marker='o')
	
	# Lines
	x = np.linspace(0, 1, 30)
	pl.plot(x, x, 'k') 
	pl.plot(x, x + 0.5, 'k')
	pl.plot(x, x + 1.0, 'k')
	pl.plot(x, x + 1.5, 'k')
	if var[1] == 'dlc': pl.plot(x, x + 2.0, 'k')
	
	# Label / save plot
	if var[1] == 'dlc': pl.legend((s1, s2, s3, s4, s5), (lab1[:5]), loc='best')
	if var[1] == 'olc': pl.legend((s1, s2, s3, s4), (lab4[:4]), loc='best')
	pl.xlabel('Input Parameter Values')
	pl.ylabel('Output Parameter Values')
	pl.xlim(0, 1) 
	if var[1] == 'dlc': pl.ylim(0, 3.5)
	if var[1] == 'olc': pl.ylim(0, 3.0)
	pl.savefig(var[6] + name)

# Plot learning curve
def plot_learning(train, gens, name):
	print '\n  Plotting learning curve...\n'
	
	# Iteration array
	iters = np.linspace(1, len(train), len(train))
	
	# Plot
	pl.figure('cost', figsize = (6, 4))
	pl.plot(np.log10(iters), train/gens, c='r')
	pl.xlabel('Training Iterations (log scale)')
	pl.ylabel('Cost Function / Exemplar')
	pl.savefig(var[6] + name)

# Edit settings
if mode == 0:
	new_var = []

	# Ask for new settings
	for i in range(0, len(var)):
		print '\n  \033[1;31mSetting %i/%i\033[1;m:' %(i+1, len(var)) + des[i].split('=')[-1]
		print '    Current value: %s' %var[i]
		user = raw_input('    (1) Change, (2) keep, or (any key) exit?: ')
		if user == '1': new_var.append(raw_input('    Input new value: '))
		if user != '1': new_var.append(var[i])
		if user != '1' and user != '2': break

	# Pad new settings array if exited
	if len(new_var) != 10:
		gap = len(var) - len(new_var)
		size = len(new_var)
		for i in range(0, gap):
			new_var.append(var[i+size])

	# Save new settings
	var, descrip = make_settings(values = new_var)

# Compute & save synthetic LCs - Prsa 2008
if mode == 1:
	# Estimated time: num LCs / num jobs * 10 secs
	
	# Slurm access only
	if len(sys.argv) >= 3:
		if sys.argv[2] == 'slurm':

			# Run subprocess
			if sys.argv[3] == 'subprocess':
				compute_lcs(start = int(sys.argv[5]), end = int(sys.argv[5])+1)
				exit()

			# Define start and end points
			start, end = int(sys.argv[4]), int(sys.argv[5])
			
			# Progress bar
			print '\n  Computing and saving theoretical lightcurves...\n'
			progress(start-int(sys.argv[4]), end-int(sys.argv[4]))

			# Initiate subprocesses
			while(start < end):
				cmd = 'python %s %s slurm subprocess %s %i' %(sys.argv[0], sys.argv[1], sys.argv[3], start)
				proc = subprocess.Popen(splitsh(cmd)) #, stdout=open(os.devnull, 'w')) << adding this prevents file saves
				#proc = subprocess.Popen(splitsh(cmd), stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)#
				time.sleep(15)			# Initiate 15 sec wait, can not do less than 15 second timeout without error
				pstatus = proc.poll()	# Check on status
				if pstatus == None: 	# None = ongoing, 0 = done
					print '\n\n\n  ERROR: SUBPROCESS TIMED OUT. KILLING...\n\n'
					proc.kill()			# Kill process 
					#os.killpg(os.getpgid(proc.pid), signal.SIGTERM)  # Send the signal to all the process groups	
				else: 					# Inform of success
					start = start + 1
				progress(start-int(sys.argv[4]), end-int(sys.argv[4]))
				if start == end: exit()
			exit() # End program
	
	# Ask for number of LCs to produce
	num = input('\033[1;34m  Number of LCs to compute: \033[1;m')
	njobs = 100
	step = int(num / njobs)
	
	# Create and submit njobs slurm jobs
	for x in range(1, njobs+1):
		command = 'python ebai.py %i slurm %s %i %i' %(mode, var[1], step*(x-1), step*x)
		name = 'ebai.%i.%s.%i' %(mode, var[1], x)
		make_job(command, job_name = name)
		os.system('sbatch %s%s.sh' %(var[5], name))

	# Make list file for output
	make_file(['%i.' %x + var[1] for x in range(0, num)], save_dir = var[3], save_file = 'list.%s.txt' %var[1])
	
# Load, plot, save paramater distributions - Hause 2016
if mode == 2:
	# Estimated 20 sec / 50,000 LCs
	
	# Slurm access only
	if len(sys.argv) >= 3:
		if sys.argv[2] == 'slurm':
			plot_params(list_file = var[3] + 'list.%s.txt' %var[1], bounds_save = var[3] + 'bounds.%s.txt' %var[1])
			exit() # End program

	# Create and submit slurm job
	command = 'python ebai.py %i slurm' %mode
	name = 'ebai.%i.%s' %(mode, var[1])
	make_job(command, job_name = name)
	os.system('sbatch %s%s.sh' %(var[5], name))

# Polyfit LCs - Prsa 2008
if mode == 3:
	# Estimated time: num LCs / num jobs * 0.7 sec/LC 
	
	# Slurm access only
	if len(sys.argv) >= 3:
		if sys.argv[2] == 'slurm':
			polyfit_lcs(list_file = var[3] + 'list.%s.txt' %var[1], njobs = int(sys.argv[3]))
			exit() # End program
	
	# Create and submit njobs slurm jobs
	njobs = 100
	for x in range(1, njobs+1):
		command = 'python ebai.py %i slurm %i %i' %(mode, njobs, x)
		name = 'ebai.%i.%s.%i' %(mode, var[1], x)
		make_job(command, job_name = name)
		os.system('sbatch %s%s.sh' %(var[5], name))
	
	# Make list file for output
	num = len(np.loadtxt(var[3] + 'list.%s.txt' %var[1], dtype=str))
	make_file(['%i.pf.' %x + var[1] for x in range(num)], save_dir = var[3], save_file = 'list.pf.%s.txt' %var[1])
	
# Reformat polyfitted LCs for ANN processing - Hause 2016
if mode == 4 or mode == 5:
	# Synthetic: Estimated 30 mins / 50,000 LCs
	# Real: Estimated 3 mins / 3,000 LCs

	# Slurm access only
	if len(sys.argv) >= 3:
		if sys.argv[2] == 'slurm':
			reformat_lcs()
			exit() # End program

	# Create and submit slurm job
	if mode == 4:
		command = 'python ebai.py %i slurm synthetic' %mode
		name = 'ebai.%i.syn' %mode
	if mode == 5:
		command = 'python ebai.py %i slurm real' %mode
		name = 'ebai.%i.real' %mode

	make_job(command, job_name = name)
	os.system('sbatch %s%s.sh' %(var[5], name))

# Identify and separate detached / overcontact EBs - Hause 2016
if mode == 6: 
	# Estimated 1 hr for 50,000 + 3,000 LCs
	# Should only be run for a least of couple thousands LCs
	
	# Slurm access only
	if len(sys.argv) >= 3:
		if sys.argv[2] == 'slurm':
			reduce_lcs()
			exit() # End program

	# Create and submit slurm job
	command = 'python ebai.py %i slurm real' %mode
	name = 'ebai.%i.real' %mode
	make_job(command, job_name = name)
	os.system('sbatch %s%s.sh' %(var[5], name))

# Chi-based reduction of synthetic LC distribution - Hause 2016
if mode == 7: 
	# Estimated 1.5 hrs for 50,000 + 2,000 LCs
	# Should only be run for a least of couple thousands LCs
	
	# Slurm access only
	if len(sys.argv) >= 3:
		if sys.argv[2] == 'slurm':
			reduce_lcs()
			exit() # End program

	# Create and submit slurm job
	command = 'python ebai.py %i slurm synthetic' %mode
	name = 'ebai.%i.%s' %(mode, var[1])
	make_job(command, job_name = name)
	os.system('sbatch %s%s.sh' %(var[5], name))

# Train ANN - Prsa 2008
if mode in [8, 10, 11, 12]:	

	# Set ANN arguments
	iters = 3000000 # Number of total training iterations
	path = os.getcwd() + '/'
	opdir = path+var[8]+'weights/'+var[1]+'/' # not used
	weights = ' --i2h i2h.%s.weights --h2o h2o.%s.weights' %(var[1], var[1])
	lcs_folder = path + var[3]
	num_lcs = len(np.loadtxt(lcs_folder + 'list.ann.par.pf.%s.txt' %var[1], dtype=str))
	lcs_dir = '--data-dir %s' %(lcs_folder[:-1])
	format = '--data-format %d.ann.par.pf.' + var[1]
	bounds = '--param-bounds %sbounds.ann.par.pf.%s.txt' %(lcs_folder, var[1])
	
	# Number of principal parameters
	if var[1] == 'dlc': pp = 5
	if var[1] == 'olc': pp = 4

	# Number of processors to use (~250 LCs / processor)
	procs = int(num_lcs/250) + (8 - int(num_lcs/250) % 8)

	# Train ANN with varying LRP & topologies - Prsa 2008
	if mode == 8:

		# Parameter ranges
		hidden = range(5, 81, 5)
		lrp = [0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0]

		# Make output names and save list
		save_arr = ['%i.hidden.%s.txt' %(x, var[1]) for x in hidden]
		make_file(save_arr, save_dir = var[8], save_file = 'list.hidden.%s.txt' %var[1])
		
		# Create and submit slurm job (hidden layers)
		command = ['mpirun ebai.mpi -t -i 10000 -s %i -n %i:%i:%i --lrp 0.01 %s %s %s' %(num_lcs, int(var[0])+1, x, pp, lcs_dir, format, bounds) for x in hidden] 
		name = 'ebai.%i.hid.%s' %(mode, var[1])
		make_job(command, job_name = name, output = [path+var[8]+save_arr[i] for i in range(len(save_arr))], processors = procs)#, opdir = opdir)
		#os.system('sbatch %s%s.sh' %(var[5], name))
		
		# Make output names and save list
		save_arr = ['%s.lrp.%s.txt' %(str(x), var[1]) for x in lrp]
		make_file(save_arr, save_dir = var[8], save_file = 'list.lrp.%s.txt' %var[1])
		
		# Create and submit slurm job (learning rate)
		command = ['mpirun ebai.mpi -t -i 100 -s %i -n %i:40:%i --lrp %s %s %s %s' %(num_lcs, int(var[0])+1, pp, str(x), lcs_dir, format, bounds) for x in lrp]
		name = 'ebai.%i.lrp.%s' %(mode, var[1])
		make_job(command, job_name = name, output = [path+var[8]+save_arr[i] for i in range(len(save_arr))], processors = procs)#, opdir = opdir)
		os.system('sbatch %s%s.sh' %(var[5], name))

	# Initialize ANN training - Prsa 2008
	if mode == 10:
		
		# Create and submit slurm job
		command = 'mpirun ebai.mpi -t -i %i -n %i:40:%i -s %i --lrp 0.05 %s %s %s' %(iters, int(var[0])+1, pp, num_lcs, lcs_dir, format, bounds)
		name = 'ebai.%i.%s' %(mode, var[1])
		save_file = 'train.%s.%ik.%ik.txt' %(var[1], iters/1000, num_lcs/1000)
		make_job(command, job_name = name, output = var[8] + save_file, processors = procs)#, opdir = opdir)
		os.system('sbatch %s%s.sh' %(var[5], name))

	# Continued ANN training - Prsa 2008
	if mode == 11:
		
		# Create and submit slurm job
		command = 'mpirun ebai.mpi -c -i %i -n %i:40:%i -s %i --lrp 0.01 %s %s %s' %(iters/2, int(var[0])+1, pp, num_lcs, lcs_dir, format, bounds)
		name = 'ebai.%i.%s' %(mode, var[1])
		save_file = 'train.%s.%ik.%ik.txt' %(var[1], iters/1000, num_lcs/1000)
		make_job(command + weights, job_name = name, output = var[8] + save_file, processors = procs)#, opdir = opdir)
		os.system('sbatch %s%s.sh' %(var[5], name))

	# Run ANN recognition and process results - Prsa 2008
	if mode == 12:
		
		# Slurm access only
		if len(sys.argv) >= 3:
			if sys.argv[2] == 'slurm':
				
				# Import EBAI training
				print '\n  Importing training file...'
				gtag = '%s.%ik.%ik' %(var[1], iters/1000, num_lcs/1000) # Synthetic plot tag
				train = np.loadtxt(var[8]+'train.%s.txt' %gtag, dtype=np.float64).T[1]
				#gtag = '%s.%ik.%ik' %(var[1], iters/1000, num_lcs/1000) # Synthetic plot tag
				#train = np.append(train, np.loadtxt(var[8]+'train.%s.txt' %gtag, dtype=np.float64).T[1])
				
				lcs_folder = var[4]
				num_lcs = len(np.loadtxt(lcs_folder + 'list.ann.pf.%s.txt' %var[1], dtype=str))				
				ktag = '%s.%ik.%ik' %(var[1], iters/1000, num_lcs/1000) # Real plot tag

				# Import EBAI results
				print '\n  Importing recognition files...'
				gen_recog = np.loadtxt(var[8]+'recog.syn.%s.txt' %gtag, dtype=np.float64)
				kep_recog = np.loadtxt(var[8]+'recog.kep.%s.txt' %ktag, dtype=np.float64)
				
				# Plot parameter distributions
				print '\n  Plotting principal parameter distributions...'
				plot_dist(kep_recog.T[0], lab1[0], 'alpha.kep.%s.png' %ktag)
				plot_dist(kep_recog.T[2], lab1[1], 'beta.kep.%s.png' %ktag)
				plot_dist(kep_recog.T[4], lab1[2], 'gamma.kep.%s.png' %ktag)
				plot_dist(kep_recog.T[6], lab1[3], 'delta.kep.%s.png' %ktag)
				if var[1] == 'dlc': plot_dist(kep_recog.T[8], lab1[4], 'epsilon.kep.%s.png' %ktag)
				
				# Scatter plots
				plot_mapping(genN_recog, 'i2o.syn.%s.png' %gtag)
				plot_learning(train, len(gen_recog.T[0]), 'learning.%s.png' %gtag)
				
				# Lightcurve file names
				if var[1] == 'olc': files = np.loadtxt(var[3]+'accepted.0.1%.5079.gens.olc.txt', dtype=str)
				if var[1] == 'dlc': files = np.loadtxt(var[3]+'accepted.0.1%.15828.gens.dlc.txt', dtype=str)
				pars = load_params(files)[pp:] # Get exemplar principal parameters

				# Compute errors
				print '\n\n  Computing errors between parameter distributions...'
				for a in range(0, len(pars)):
					error = [(pars[a][x] - gen_recog.T[2*a][x]) / pars[a][x] * 100 for x in range(0, len(pars[a]))]
					
				exit() # End program
		
		command, output = [], [] # Initialize arrays
		
		# Base command frame
		frame = 'ebai -r -n %i:40:%i --lrp 0.01 %s' %(int(var[0])+1, pp, bounds)

		# Command and output (Synthetic)
		command.append(frame+weights+' -s %i %s %s' %(num_lcs, lcs_dir, format))
		output.append(var[8] + 'recog.syn.%s.%ik.%ik.txt' %(var[1], iters/1000, num_lcs/1000))

		# Command and output (real)
		lcs_folder = var[4]
		lcs_dir = '--data-dir ' + lcs_folder
		format = '--data-format %d.ann.pf.' + var[1]
		num_lcs = len(np.loadtxt(lcs_folder + 'list.ann.pf.%s.txt' %var[1], dtype=str))
		command.append(frame+weights+' -s %i %s %s --unknown-data' %(num_lcs, lcs_dir, format)) 
		output.append(var[8] + 'recog.kep.%s.%ik.%ik.txt' %(var[1], iters/1000, num_lcs/1000))

		# Create and submit slurm job
		name = 'ebai.%i.1.%s' %(mode, var[1])
		make_job(command, job_name = name, output = output)
		os.system('sbatch %s%s.sh' %(var[5], name))

		time.sleep(15) # Initiate 20 sec wait

		# Create and submit slurm job
		name = 'ebai.%i.2.%s' %(mode, var[1])
		make_job('python ebai.py %i slurm' %mode, job_name = name)
		os.system('sbatch %s%s.sh' %(var[5], name))

# Plot ANN training results from LRP & topology variations - Prsa 2008
if mode == 9:

	# Slurm access only
	if len(sys.argv) >= 3:
		if sys.argv[2] == 'slurm':
			tag = '%ik' %int(len(np.loadtxt('%slist.ann.par.pf.%s.txt' %(var[3], var[1]), dtype=str)) / 1000)
			#tag = '%ik' %int(len(np.loadtxt('%slist.par.pf.%s.txt' %(var[3], var[1]), dtype=str)) / 1000)

			# File names
			#names_hid = np.loadtxt('%slist.hidden.%s.txt' %(var[8], var[1]), dtype=str)
			names_lrp = np.loadtxt('%slist.lrp.%s.txt' %(var[8], var[1]), dtype=str)
			
			# Read-in hidden layer files 
			print '\n  Reading in hidden layer files...\n'
			#cfs1, val1 = load_train(names_hid)

			# Read-in, plot LRP files 
			print '\n\n  Reading in LRP files...\n'
			cfs2, val2 = load_train(names_lrp)
			
			# Plots
			print '\n\n  Plotting cost functions...\n'
			#plot_hidden(cfs1, val1, tag)
			plot_lrp(cfs2, val2, tag)
			
			exit() # End program

	# Create and submit slurm job
	command = 'python ebai.py %i slurm' %mode
	name = 'ebai.%i.%s' %(mode, var[1])
	make_job(command, job_name = name)
	os.system('sbatch %s%s.sh' %(var[5], name))

"""* Technical experience is required in at least some of the following areas:
* Mathematical modeling, for example in: MatLab, Predictive Workbench/SPSS or R
* Programming: Java, C++, ABAP
* Software architectures: n-tier, SOA
* Database servers, for example: SAP HANA, Sybase ASE / ASA, ORACLE, MS SQL Server, IBM DB2
* Database programming: SQL, PL/SQL, T-SQL
* Data warehousing/reporting/ETL: SAP BusinessObjects
* Microsoft Office (Excel, PowerPoint)

Quantitative Research Analyst
Technology Analyst
Programmer
Data Scientist

Vanguard
CIA
FBI
NASA
Deutsche Bank
Deloitte
JP Morgan
Black Rock

Pathways Recent Graduates Program

The Software Developer will be responsible for, but limited to:

Develop N-tier solutions using C#, ASP.NET, MVC, Entity Framework and SQL Server to deliver new features and system improvements as part of a project team
Develop and maintain APIs using ASP.NET, Web API and Entity Framework
Develop front-end using JavaScript, jQuery, JSON, HTML and modern web technologies
Perform hands-on software design and development
Ability to problem solve for server-side, database and client-side aspects of web development
Nice to have - develop cross platform mobile apps using HTML5 and hybrid technology (Cordova/PhoneGap, Telerik Kendo Mobile UI)

Java, JavaScript, ASP.NET, HTML, JSON

Programming Skills * 
ASP.NET
Java
PHP
Python
Ruby

Mobile Skills * AndroidiOSHTML5jQuery MobileNone of the aboveOther (Please specify)
Database Skills * 
MySQL
MS SQL
PostrgreSQL

Frontend Skills * CSSJavaScriptGraphic DesignNone of the aboveOther (Please speci"""

# Plots for AAS Poster
if mode == 13:

	# Set ANN arguments
	iters = 3000000 # Number of total training iterations

	# Slurm access only
	if len(sys.argv) >= 3:
		if sys.argv[2] == 'slurm':
				
			# Import dlc files
			print '\n  Importing dlc files...'
			dnum = len(np.loadtxt(var[3] + 'list.ann.par.pf.dlc.txt', dtype=str))
			dknum = len(np.loadtxt(var[4] + 'list.ann.pf.dlc.txt', dtype=str))	
			dtag = 'dlc.%ik.%ik' %(iters/1000, dnum/1000) # Synthetic plot tag
			dktag = 'dlc.%ik.%ik' %(iters/1000, dknum/1000) # Real plot tag
			dtrain = np.loadtxt(var[8]+'train.%s.txt' %dtag, dtype=np.float64).T[1]
			dlc_recog = np.loadtxt(var[8]+'recog.syn.%s.txt' %dtag, dtype=np.float64)
			dk_recog = np.loadtxt(var[8]+'recog.kep.%s.txt' %dktag, dtype=np.float64)
			dfiles = np.loadtxt(var[3]+'accepted.0.1%.15828.gens.dlc.txt', dtype=str)
			dpars = load_params(dfiles)
			dflux = load_lcs(dfiles[:10])

			# Import olc files
			print '\n  Importing olc files...'
			onum = len(np.loadtxt(var[3] + 'list.ann.par.pf.olc.txt', dtype=str))
			oknum = len(np.loadtxt(var[4] + 'list.ann.pf.olc.txt', dtype=str))	
			otag = 'olc.%ik.%ik' %(iters/1000, onum/1000) # Synthetic plot tag
			oktag = 'olc.%ik.%ik' %(iters/1000, oknum/1000) # Real plot tag
			otrain = np.loadtxt(var[8]+'train.%s.txt' %otag, dtype=np.float64).T[1]
			olc_recog = np.loadtxt(var[8]+'recog.syn.%s.txt' %otag, dtype=np.float64)
			ok_recog = np.loadtxt(var[8]+'recog.kep.%s.txt' %oktag, dtype=np.float64)
			ofiles = np.loadtxt(var[3]+'accepted.0.1%.5079.gens.olc.txt', dtype=str)
			opars = load_params(ofiles)
			oflux = load_lcs(ofiles[:10])
					
			# Plot distribution
			npars = []
			def plot_2dist(vals1, vals2, label, save_file, bounds):
				#if 'dlc' in ptag: pl.title('Detached EBs - %s' %label)
				#if 'olc' in ptag: pl.title('Overcontact EBs - %s' %label)
				# Unnormalize Values
				#vals1 = [bounds[0] + (vals1[x]-0.1)/(0.9-0.1)*(bounds[1]-bounds[0]) for x in range(0, len(vals1))]
				vals2 = [bounds[0] + (vals2[x]-0.1)/(0.9-0.1)*(bounds[1]-bounds[0]) for x in range(0, len(vals2))]
				npars.append(vals2)

				w1 = np.ones_like(vals1)/len(vals1)
				w2 = np.ones_like(vals2)/len(vals2)
				rang = max(max(vals1), max(vals2)) - min(min(vals1), min(vals2))
				rang1 = max(vals1) - min(vals1)
				rang2 = max(vals2) - min(vals2)
				b1 = rang1/rang * 40
				b2 = rang2/rang * 40
				binwidth = rang/50.0
				bin = np.arange(min(min(vals1), min(vals2)), max(max(vals1), max(vals2)) + binwidth, binwidth)
				f = pl.figure(save_file, figsize = (6, 5))
				h = pl.hist(vals1, bins=bin, weights=w1*100, histtype='bar', align='mid', orientation='vertical', label='Computed')
				j = pl.hist(vals2, bins=bin, weights=w2*100, histtype='bar', align='mid', orientation='vertical', label='Kepler')	
				pl.legend(loc='best')
				pl.xlabel('%s' %label); pl.ylabel('Frequency [%]')
				#pl.xlim([0, 1])
				pl.savefig(var[6] + 'parameters/' + save_file)
				pl.close(f)

			# Plot parameter distributions - dlc
			print '\n  Plotting dlc parameter distributions...'
			ptag = 'dlc.%i.kep-v-gen.png' %(iters/1000)
			bounds = np.loadtxt(var[3]+'bounds.ann.par.pf.dlc.txt', dtype=np.float64)
			plot_2dist(dpars[0], dk_recog.T[0], lab1[0], 'alpha.%s' %ptag, bounds[0])
			plot_2dist(dpars[1], dk_recog.T[2], lab1[1], 'beta.%s' %ptag, bounds[1])
			plot_2dist(dpars[2], dk_recog.T[4], lab1[2], 'gamma.%s' %ptag, bounds[2])
			plot_2dist(dpars[3], dk_recog.T[6], lab1[3], 'delta.%s' %ptag, bounds[3])
			plot_2dist(dpars[4], dk_recog.T[8], lab1[4], 'epsilon.%s' %ptag, bounds[4])

			# Save parameter solutions
			def save_solutions(file, id):
				for y in range(0, len(id)):
					file.write("%s\t" %id[y].split('/')[-1])
					for x in range(0, len(npars)):	
						file.write("%f\t" %npars[x][y])
					file.write("\n")
				file.close()

			# Save dlc parameter solutions
			file = open("solutions.dlc.txt", 'w')
			id = np.loadtxt(var[4] + '1.5%.2109.ids.dlc.txt', dtype=str)	
			save_solutions(file, id)

			# Plot parameter distributions - olc
			npars = []
			print '\n  Plotting olc parameter distributions...'
			ptag = 'olc.%i.kep-v-gen.png' %(iters/1000)
			bounds = np.loadtxt(var[3]+'bounds.ann.par.pf.olc.txt', dtype=np.float64)
			plot_2dist(opars[0], ok_recog.T[0], lab4[0], 'alpha.%s' %ptag, bounds[0])
			plot_2dist(opars[1], ok_recog.T[2], lab4[1], 'beta.%s' %ptag, bounds[1])
			plot_2dist(opars[2], ok_recog.T[4], lab4[2], 'gamma.%s' %ptag, bounds[2])
			plot_2dist(opars[3], ok_recog.T[6], lab4[3], 'delta.%s' %ptag, bounds[3])

			# Save olc parameter solutions
			file = open("solutions.olc.txt", 'w')
			id = np.loadtxt(var[4] + '1.5%.766.ids.olc.txt', dtype=str)
			save_solutions(file, id)

			# Plot learning curve
			def plot_2learning(train1, train2, gens1, gens2, name):
				
				# Iteration array
				iters1 = np.linspace(1, len(train1), len(train1))
				iters2 = np.linspace(1, len(train2), len(train2))
				
				# Plot
				pl.figure('cost', figsize = (8, 4))
				pl.plot(np.log10(iters1), train1/gens1, c='b', label = 'Detached')
				pl.plot(np.log10(iters2), train2/gens2, c='r', label = 'Overcontact')
				pl.xlabel('Training Iterations (log scale)')
				pl.ylabel('Cost Function / Exemplar')
				pl.legend(loc='best')
				pl.savefig(var[6] + name)

			# Scatter plots
			print '\n  Plotting learning curve...\n'
			plot_2learning(dtrain, otrain, len(dlc_recog.T[0]), len(olc_recog.T[0]), 'learning.%ik.dlc-v-olc.png' %(iters/1000))
			var[1]='dlc'; plot_mapping(dlc_recog, 'i2o.syn.%s.png' %dtag)
			var[1]='olc'; plot_mapping(olc_recog, 'i2o.syn.%s.png' %otag)

			# Plot lightcurve
			def plot_lcs(flux, name):
				phases = np.linspace(-0.5, 0.5, int(var[0]))
				f = pl.figure(name, figsize = (6, 4))
				pl.plot(phases, flux, c='r')
				pl.xlabel('Phase')
				pl.ylabel('Normalized Flux')
				pl.savefig(var[6] + name)
				pl.close(f)

			for x in range(0, 5, 1):
				plot_lcs(dflux[x], 'dlc.%i.png' %x)
				plot_lcs(oflux[x], 'olc.%i.png' %x)
			
			exit() # End program

	# Create and submit slurm job
	name = 'ebai.%i' %mode
	make_job('python ebai.py %i slurm' %mode, job_name = name)
	os.system('sbatch %s%s.sh' %(var[5], name))