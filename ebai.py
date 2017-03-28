# Eclipsing Binaries via Artificial Intelligence - Hause, Prsa et al. 2017

import numpy as np                      # Basic math, arrays
import sys, os, glob                    # Interaction with system
import re								# For reading variables from file names
from collections import Counter			# For chi tests
import matplotlib                       # Basic plotting
matplotlib.use('Agg')                   # For remote use on server
import matplotlib.pyplot as pl          # Basic plotting
import subprocess, time                 # For subprocessing
from itertools import islice			# For reading parameters from files
from shlex import split as splitsh      # ^^
import phoebeBackend as pb              # PHOEBE, eclipsing binary (EB) modeling engine
from numpy.random import uniform as rng # For Monte-Carlo parameter generation

# EBAI setup - settings and directories 
def ebai_setup(key, preset = None):

	# Create default settings file
	def make_settings(fname, values = [None, '500', None, None, None, None, None]):
		file = open(fname, 'w')

		if values[0] != None: 
			file.write('%s= Current real EB workset, name of directory to work in\n' %values[0])
		else:
			dir = raw_input('\n\tReal EB data set name: ')
			file.write('%s= Current real EB workset, name of directory to work in\n' %dir)

		file.write('%s= Data points per lightcurve\n' %values[1])
		file.write('%s= Email for slurm jobs\n' %values[2])
		file.write('%s= Fastest converging ANN learning rate (detached)\n' %values[3])
		file.write('%s= Fastest converging ANN learning rate (contact)\n' %values[4])
		file.write('%s= Fastest converging ANN topology (detached)\n' %values[5])
		file.write('%s= Fastest converging ANN topology (contact)\n' %values[6])
		file.close()
			
		return np.genfromtxt(fname, delimiter='=', dtype=str).T

	# Main function
	def main(key, preset):
		
		# Reading / creating mode
		if key == 'r':

			try: 	# Check for settings file to load
				sett, des = np.genfromtxt('ebai.settings', delimiter='=', dtype=str).T
			except: # Create default settings file if none found
				try: sett, des = np.genfromtxt('../ebai.settings', delimiter='=', dtype=str).T
				except: sett, des = make_settings('ebai.settings')

			# Check for / make directories
			def make_dir(dir):
				try: os.mkdir(dir)
				except: pass

			# Make directories if they don't exist
			if 'slurm' not in sys.argv and 'subprocess' not in sys.argv:
				make_dir('model_lightcurves/'); make_dir('phoebe/'); 
				make_dir(sett[0])
				os.chdir(sett[0]) # Change to working directory
				make_dir('lightcurves/'); make_dir('jobs/')
				make_dir('plots/'); make_dir('ann/')
			elif 'subprocess' not in sys.argv: 
				os.chdir(sett[0]) # Change to working directory

		# Writing / editing mode
		if key == 'w':
			if preset == None: # Manually edit settings file
				sett, des = np.genfromtxt('../ebai.settings', delimiter='=', dtype=str).T

				# Display settings
				for i in range(len(sett)):
					print '\n\t(%i)%s' %(i+1, des[i])
					print '\t    Current value: %s' %sett[i]
				
				# Change settings
				for x in range(len(sett)):
					ans = input('\n\tSetting to change (0 to exit): ')
					if ans in range(1, 7):
						sett[ans-1] = raw_input('\n\tInput new value: ')
					else: break
					
				sett, des = make_settings('../ebai.settings', sett) # Save new settings
			
			else: # Write optimal ann parameters to settings file
				sett, des = make_settings('../ebai.settings', preset)

		return sett # Return settings

	sett = main(key, preset)
	return sett # Return settings

# Start-up instructions
def ebai_instructs(dir):

	if 'subprocess' not in sys.argv: 
		print '\n\n\033[1;31m [EBAI] - Eclipsing Binaries via Artificial Intelligence\033[1;m\n'

		print '\033[1;34m List of System Arguments\033[1;m\n'
		print '\033[1;34m  (0)\033[1;m\033[1;38m Edit ebai.settings file\033[1;m'
		print '\033[1;30m\t- File is automatically created if not in script directory'
		print '\033[1;34m  (1)\033[1;m\033[1;38m Compute model EB lightcurves via PHOEBE\033[1;m'
		print '\033[1;30m\t- Creates original simulation and polyfit-ed files for each lightcurve'
		print '\033[1;34m  (2)\033[1;m\033[1;38m Polyfit real EB lightcurves\033[1;m'
		print '\033[1;30m\t- Reads in real EBs from %s/lightcurves/' %dir
		print '\033[1;30m\t- Must provide list of file names (column 1) and object ID (column 2) in %s/' %dir
		print '\033[1;30m\t- All real EB lightcurves must be formatted as phases (column 1) and fluxes (column 2)'
		print '\033[1;34m  (3)\033[1;m\033[1;38m Chi^2 Statistical Testing\033[1;m'
		print '\033[1;30m\t- Classifies real EBs as either contact or detached\033[1;m'
		print '\033[1;30m\t- Verifies geometric overlap between computed and real EBs\033[1;m'
		print '\033[1;30m\t- Reformats all lightcurve files to be ANN-readable\033[1;m'
		print '\033[1;34m  (4)\033[1;m\033[1;38m Optimize ANN parameters'
		print '\033[1;30m\t- Trains ANN with various learning rate parameters & topologies\033[1;m'
		print '\033[1;30m\t- Plots residuals as function of these variations to identify optimal parameters\033[1;m'
		print '\033[1;34m  (5)\033[1;m\033[1;38m Initialize ANN training\033[1;m'
		print '\033[1;34m  (6)\033[1;m\033[1;38m Run ANN recognition mode and analyze results\033[1;m'
	
		if len(sys.argv) == 1: # End if user provided no system argument
			exit() 
		if sys.argv[1] not in map(str, range(7)): # Error message
			print '\n\033[1;31m  ERROR:\033[1;m\033[1;38m Invalid user input\n\n'
			exit()
		
		print '\n\033[1;34m  Mode: \033[1;m\033[1;38m' + sys.argv[1]

	return int(sys.argv[1]) # Return mode selection
####
sett = ebai_setup('r')
mode = ebai_instructs(sett[0])
####
# Progess bar function
def progress(done, total):
	barLen, progress = 25, ''
	for i in range(barLen):
		if (i < int(barLen * done / total)): progress += '>'
		else: progress += ' '
	sys.stdout.write('\r\t\033[1;32m    [%s]\033[1;m' %progress)
	sys.stdout.write('\033[1;38m - \033[1;m')
	sys.stdout.write('\033[1;31m%s\033[1;m' %str(done))
	sys.stdout.write('\033[1;38m of \033[1;m')
	sys.stdout.write('\033[1;31m%s\033[1;m' %str(total))
	sys.stdout.write('\033[1;38m processed (\033[1;m')
	sys.stdout.write('\033[1;31m%.2f%%\033[1;m' %(done * 100. / total))
	sys.stdout.write('\033[1;38m)\033[1;m')
	sys.stdout.flush()

# Create sbatch job
def run_job(command, job_name, output = 0, processors = 1):
	job = open('%s.sh' %job_name, 'w')
	job.write('#!/bin/bash\n')
	job.write('\n#SBATCH -J %s' %job_name.split('/')[-1])
	job.write('\n#SBATCH -p big')
	job.write('\n#SBATCH -N 1')
	job.write('\n#SBATCH -n %i' %processors)
	job.write('\n#SBATCH -t 2-00:00:00')
	job.write('\n#SBATCH -o %s/%s.out' %(sett[0], job_name))
	job.write('\n#SBATCH -D %s\n\n' %os.getcwd().replace('/'+sett[0], ''))
	if '@' in sett[2]:
		job.write('#SBATCH --mail-type=BEGIN,END,FAIL')
		job.write('\n#SBATCH --mail-user=%s\n\n' %sett[2])
	
	# Write jobs to run
	if isinstance(command, list) == True: # For list of commands
		for x in range(len(command)):
			job.write(command[x])
			try: job.write(' > '+ output[x] + '\n')
			except: job.write('\n')
	else: # For singular command
		job.write(command)
		if output != 0: job.write(' > '+ output)
	
	job.close() 						# Close file
	os.system('sbatch %s.sh' %job_name) # Run job

# Plot parameter distributions
def plot_params(pars, save_dir, tag):
	print '\n\tPlotting parameter distributions...'
	save_par = ['alpha.', 'beta.', 'gamma.', 'delta.', 'epsilon.']
	label = [[r'$T_2/T_1$']*2, [r'$\rho_1 + \rho_2$', r'$M_2/M_1$'], 
			[r'$e sin(\omega)$', r'$(\Omega^I-\Omega)/(\Omega^I-\Omega^O)$'],
			[r'$e cos(\omega)$', r'$sin(i)$'], [r'$sin(i)$']*2]
	i = 0 if 'dlc' in tag else 1
	for x in range(len(pars)):
		f = pl.figure(save_par[x], figsize = (6, 4))
		weight = np.ones_like(pars[x])/len(pars[x])
		h = pl.hist(pars[x], bins=50, weights=weight*100, histtype='bar', align='mid', orientation='vertical', label=label[x][i])
		pl.legend(loc='best')
		pl.xlabel('Value'); pl.ylabel('Frequency (%)')
		pl.xlim([min(pars[x]), max(pars[x])])
		pl.savefig(save_dir + save_par[x] + tag)
		pl.close(f)

# Polyfit lightcurve	
def polyfit(in_file, out_file, pf_order = 2, iters = 10000):
	command = 'polyfit -o %i -i %i -n %s -c 0 1 --find-knots --find-step ' %(pf_order, iters, sett[1])
	os.system(command + in_file + ' > ' + out_file)

# Compute and save synthetic LCs
def compute_lightcurves():

	# Create slurm job for main
	def setup():
		if 'slurm' not in sys.argv and 'subprocess' not in sys.argv:
			for eb_type in ['dlc', 'clc']:

				# Ask for number of LCs to compute
				eb = 'detached' if 'dlc' in eb_type else 'contact'
				num = input('\n\033[1;34m\tNumber of %s EBs to compute (0 for none): \033[1;m' %eb)
				if num != 0: 
					jobs = input('\033[1;34m\tNumber of simultaneous %s-computing jobs: \033[1;m' %eb)
				else: continue
				
				# Create and submit 'jobs' slurm jobs
				step = int(num/jobs)
				[run_job('python ebai.py 1 slurm %s %i %i' %(eb_type, step*y, step*(y+1)), 'jobs/ebai.1.%s.%i' %(eb_type,y)) for y in range(jobs)]
			exit() # End program

	# Compute lightcurve
	def create_lightcurve(x, eb_type):
	
		# Get random principal parameters
		def principal_params():
			if 'dlc' in eb_type:

				# Alpha principal parameter = T2/T1: roughly the surface brightness ratio
				alpha = 1-abs(0.18*np.sqrt(-2*np.log(rng(0,1)))*np.cos(2*np.pi*rng(0,1))) 

				beta = 0.05 + rng(0,0.45)				# Beta principal parameter = rho1 + rho2: Fractional radii sum of both stars
				e0max = 0.5*np.exp(-6*(beta-0.05))		# Attenuation factor (dependence of eccentricity on the sum of radii):
				ecc = e0max * -1/3 * np.log(rng(0,1))	# Eccentricity
				omega = rng(0, 2*np.pi)					# Argument of periastron
				gamma = ecc * np.sin(omega)             # Gamma principal parameter
				delta = ecc * np.cos(omega)             # Delta principal parameter

				i_eclipse = np.arcsin(np.sqrt(1-(0.9*beta)**2))     
				incl = i_eclipse + rng(0,(np.pi/2)-i_eclipse)      # Inclination
				epsilon = np.sin(incl)                             # Epsilon principal parameter

				# IMPORTANT: Because of the numeric instability of Omega(q) for small q,
				# the roles of stars are changed here: q > 1, star 1 is smaller and cooler,
				# star 2 is hotter and larger.

				return [alpha, beta, gamma, delta, epsilon, ecc, omega, incl] # Return the acquired random set

			elif 'clc' in eb_type:

				# Alpha principal parameter = T2/T1: roughly the surface brightness ratio
				alpha = 1/(1-abs(0.14*np.sqrt(-2*np.log(rng(0,1)))*np.cos(2*np.pi*rng(0,1)))) 

				# Beta principal parameter = mass ratio:
				beta = 1/(1-0.22*abs(np.sqrt(-2*np.log(rng(0,1)))*np.cos(2*np.pi*rng(0,1))))

				# Potentials
				potL = critical_pot(beta, 1.0, 0.0)
				pot = potL[1] + rng(0, potL[0]-potL[1])
				
				# Gamma principal parameter = fillout factor: (Omega(L1)-Omega)/(Omega(L1)-Omega(L2))
				gamma = (potL[0]-pot)/(potL[0]-potL[1])

				delta = 0.2 + rng(0,0.8)    # Delta principal parameter = sin(i)
				ecc = 0.0                   # Eccentricity
				omega = np.pi/2             # Argument of periastron

				return [alpha, beta, gamma, delta, pot, ecc, omega] # Return the acquired random set

		# Get random surface brightness parameters
		def sb_pars(pars):
			if 'dlc' in eb_type:

				# rho1:
				# this parameter is obtained by sampling from the [0.025,beta-0.025]
				# interval. The interval assures that the size of either star is not
				# smaller than 0.025 of the semimajor axis.

				r2r1 = 1+0.25*np.sqrt(-2*np.log(rng(0,1)))*np.cos(2*np.pi*rng(0,1))
				rho1 = pars[1]/(1+r2r1)
				rho2 = pars[1]*r2r1/(1+r2r1)

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

				return rho1, rho2, T1, T2

			elif 'clc' in eb_type:
				T1 = 0
				while(T1 < 3500):
					T2 = 3500 + rng(0,3500)
					T1 = T2/pars[0]

				return pb.getpar('phoebe_radius1')/10., pb.getpar('phoebe_radius2')/10., T1, T2

		# Calculate effective potential (Omega_1) of the primary star or (Omega_2) of the secondary star
		def potential(star, D, q, r, F, lmbda, nu):
			def primary_pot(D, q, r, F, lmbda, nu):
				return 1/r + q*(D**2 + r**2 -2*r*lmbda*D)**(-1/2) - r*lmbda/D**2 + (1/2)*(F**2)*(1+q)*(r**2)*(1-nu**2)
		
			if star == 1: return primary_pot(D, q, r, F, lmbda, nu)
			if star == 2: return primary_pot(D, 1./q, r, F, lmbda, nu)*q + (1/2)*(q-1)*q

			#   D      .. instantaneous separation between components in units of semi-major axis (a)
			#   q      .. mass ratio (secondary over primary)
			#   r      .. star radius in units of semi-major axis (a)
			#   F      .. synchronicity parameter
			#   lambda .. direction cosine
			#   nu     .. direction cosine

		# Calculate critical effective potentials of both stars (through L1 and L2)
		def critical_pot(q, F, e):

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
		def default_pars(pars, sbpars):

			if 'dlc' in eb_type:
				pb.setpar('phoebe_ecc', pars[5])
				pb.setpar('phoebe_perr0', pars[6])
				pb.setpar('phoebe_incl', 180*np.arcsin(pars[4])/np.pi)
				pb.setpar('phoebe_pot1', potential(1, 1-pars[5], 1, sbpars[0], 1, 1, 0))
				pb.setpar('phoebe_pot2', potential(2, 1-pars[5], 1, sbpars[1], 1, 1, 0))
				pb.setpar('phoebe_teff1', sbpars[2])
				pb.setpar('phoebe_teff2', sbpars[3])
				pb.setpar('phoebe_pshift', -1*conjunction_phase(pars[5], pars[6]))
				#pb.setpar('phoebe_lc_filter', 'Kepler:mean', 0)

			elif 'clc' in eb_type:
				pb.setpar('phoebe_incl', 180*np.arcsin(pars[3])/np.pi)
				pb.setpar('phoebe_rm', pars[1])
				pb.setpar('phoebe_pot1', pars[4])
				pb.setpar('phoebe_pot2', pars[4])
				pb.setpar('phoebe_teff1', sbpars[2])
				pb.setpar('phoebe_teff2', sbpars[3])
				pb.setpar('phoebe_pshift', 0.5)
				#pb.setpar('phoebe_lc_filter', 'Kepler:mean', 0)

		# Set value of gravity darkening based on temperature
		def grav_darkening(sbpars):

			if 'dlc' in eb_type: lim = 7500  # changes discretely at T=7500K
			elif 'clc' in eb_type: lim = 7000  # changes discretely at T=7000K

			if (sbpars[2] < lim):
				pb.setpar('phoebe_grb1', 0.32)	 # 0.32 for convective envelopes
			else: pb.setpar('phoebe_grb1', 1.00) # 1.0 for radiative envelopes

			if (sbpars[3] < lim):
				pb.setpar('phoebe_grb2', 0.32)	 # 0.32 for convective envelopes
			else: pb.setpar('phoebe_grb2', 1.00) # 1.0 for radiative envelopes

		# Verify parameters feasibility
		def feasible(pars, sbpars):

			if 'dlc' in eb_type:

				# Calculate potentials
				potC = critical_pot(1, 1, pars[5])
				pot1 = potential(1, 1-pars[5], 1, sbpars[0], 1, 1, 0)
				pot2 = potential(2, 1-pars[5], 1, sbpars[1], 1, 1, 0)

				# Determine feasibility
				#   Test 1: Is T2/T1 less than 0.2?
				#   Test 2: Is eccentricity more than 0.8?
				#   Test 3: (1.5 is arbitrary) pars[1] sum of radii
				#   Test 4: Are periastron potentials overflowing the lobe?

				if(pars[0] < 0.2 or pars[5] > 0.8 or 1.5*pars[1] > 1-pars[5] or pot1 < potC[0] or pot2 < potC[0]): return 'false'
				else: return 'true'

			if 'clc' in eb_type:

				# Determine feasibility
				#   Test 1: --?
				#   Test 2: --?

				if (pars[1] < 0.15 or pars[1] > 1/0.15): return 'false'
				elif (np.arcsin(pars[3]) < np.arccos(sbpars[0]+sbpars[1])): return 'false'
				else: return 'true'

		pb.init(); pb.configure()   # Startup phoebe
		if 'dlc' in eb_type: 	    # Open a generic detached EB model
			pb.open('../phoebe/detached.phoebe')
		elif 'clc' in eb_type: 	    # Open a generic contact EB model
			pb.open('../phoebe/contact.phoebe')
		
		# Phase tuple [-0.5, 0.5]
		phases = tuple(np.linspace(-0.5, 0.5, int(sett[1])).tolist())

		pars = principal_params()				# Get random principal parameters
		sbpars = sb_pars(pars)					# Get surface brightness parameters
		if(feasible(pars, sbpars) == 'true'):	# Check parameter feasibility
			default_pars(pars, sbpars) 			# Load default parameters
			pb.updateLD()					    # Limb darkening [coefficients from Van Hamme (1993)]
			grav_darkening(sbpars)		  		# Gravity darkening
			flux = pb.lc(phases, 0)			    # Get flux

			# Check for PHOEBE errors
			if flux != False and True not in np.isnan(flux) and flux[0] != 0:
					
				# Check if luminosities are outside the expected interval:	
				if 'dlc' in eb_type:
					sbr1 = pb.getpar('phoebe_sbr1') # Primary luminosity
					sbr2 = pb.getpar('phoebe_sbr2')	# Secondary luminosity
					if(sbr2/sbr1 < 0.1 and sbr2 > sbr1): time.sleep(60) # Freeze subprocess on rejection, wait to be killed

				# Create LC file
				fname = '../model_lightcurves/%i.' %x + eb_type
				fileout = open(fname, 'w')
											
				# Create file header
				fileout.write('# alpha   = %f\n' %pars[0])
				fileout.write('# beta    = %f\n' %pars[1])
				fileout.write('# gamma   = %f\n' %pars[2])
				fileout.write('# delta   = %f\n' %pars[3])

				if 'dlc' in eb_type: fileout.write('# epsilon = %f\n' %pars[4])
				elif 'clc' in eb_type: fileout.write('# pot     = %f\n' %pars[4])
				
				fileout.write('# ecc     = %f\n' %pars[5])
				fileout.write('# omega   = %f\n' %(pars[6]*180/np.pi))
				fileout.write('# rho1    = %f\n' %sbpars[0])
				fileout.write('# rho2    = %f\n' %sbpars[1])
				fileout.write('# Teff1   = %f\n' %sbpars[2])
				fileout.write('# Teff2   = %f\n' %sbpars[3])
			
				if 'dlc' in eb_type:
					fileout.write('# incl.   = %f\n' %(180*np.arcsin(pars[3])/np.pi))
					fileout.write('# sbr1    = %f\n' %sbr1)
					fileout.write('# sbr2    = %f\n' %sbr2)

				elif 'clc' in eb_type:
					fileout.write('# incl.   = %f\n' %(180*np.arcsin(pars[4])/np.pi))
					fileout.write('# sbr1    = N/A\n')
					fileout.write('# sbr2    = N/A\n')

				# Compile data points array
				data = [4*np.pi/(pb.getpar('phoebe_plum1')+pb.getpar('phoebe_plum2')) * flux[j] for j in range(int(sett[1]))]

				# Write phases and fluxes to file
				[fileout.write('%s\t%s\n' %(phases[j], str(data[j]))) for j in range(int(sett[1]))]

				fileout.close(); pb.quit()                              # Close out lightcurve file and PHOEBE 
				polyfit(fname, '../model_lightcurves/%i.pf.' %x + eb_type) # Polyfit new LC file
			else: time.sleep(60) # Freeze subprocess on failure, wait to be killed

	# Main function
	def main():
		# Initiate subprocess - subprocessing protects against PHOEBE freezing
		if 'subprocess' in sys.argv:
			create_lightcurve(int(sys.argv[4]), sys.argv[3]); exit()

		# Define start and end points
		print '\n\tComputing and saving theoretical lightcurves...'
		num = int(sys.argv[4]) 							# Initialize counter
		progress(0, int(sys.argv[5])-int(sys.argv[4])) 	# Progress bar

		# Create subprocess for each LC - protects against PHOEBE freezing
		while(num <= int(sys.argv[5])):
			if os.path.isfile('../model_lightcurves/%i.%s' %(num, sys.argv[3])) == False or os.path.isfile('../model_lightcurves/%i.pf.%s' %(num, sys.argv[3])) == False:
				cmd = 'python ../%s %s subprocess %s %i' %(sys.argv[0], sys.argv[1], sys.argv[3], num)
				proc = subprocess.Popen(splitsh(cmd))
				time.sleep(20)			# 20 sec wait (< 20 second timeout can cause errors)
				if proc.poll() == None: # Check on process (None = ongoing, 0 = done)
					proc.kill()			# Kill process 
			else: 
				num += 1	        		# Success
				progress(num-int(sys.argv[4]), int(sys.argv[5])-int(sys.argv[4]))

	setup()
	main()

# Polyfit real EBs
def polyfit_realEBs():

	# Create slurm job for main
	def setup():
		if 'slurm' not in sys.argv:

			# Ask for user input
			njobs = input('\033[1;34m\n\tNumber of simultaneous polyfit jobs: \033[1;m')
			list = raw_input('\033[1;34m\tList of file names and object IDs in %s/: \033[1;m' %sett[0])

			# Check to see if list exists
			try: 
				np.loadtxt(list, dtype=str)
			except: 
				print '\n\tERROR: \'%s\' does not exist\n' %list; exit()

			# Create and submit njobs slurm jobs
			[run_job('python ebai.py 2 slurm %i %i %s' %(njobs, x, list), 'jobs/ebai.2.%i' %x) for x in range(njobs)]
			exit() # Terminate

	# Main function
	def main():
		names, ids = np.loadtxt(sys.argv[5], dtype=str).T  # Get file names & object IDs
		files = np.split(names, int(sys.argv[3]))[int(sys.argv[4])]
		names = names.tolist(); ids = ids.tolist()

		progress(0, len(files))         # Start progress
		for x in range(len(files)):     # Polyfit files
			polyfit('lightcurves/'+files[x], 'lightcurves/%i.%s.pf.rlc' %(names.index(files[x]), ids[names.index(files[x])]))
			progress(x+1, len(files))   # Update progress

	setup()
	main()

# Chi^2 tests to categorize EBs, make training sets	
def chi_tests():

	# Create slurm job for main
	def setup():
		if 'slurm' not in sys.argv: 
			run_job('python ebai.py 3 slurm', 'jobs/ebai.3'); exit()
	
	# Load lightcurve fluxes from files of format
	def load_lcs(format):
		files = list(glob.iglob(format))					# Get files of format
		progress(0, len(files))								# Start progress
		fluxes = np.zeros([len(files), int(sett[1])+1])		# Fluxes array
		for x in range(len(files)):
			with open(files[x]) as f:						# Open lightcurve file
				lc = np.zeros(int(sett[1])+1)
				for line in f:
					if not line.strip().startswith("#"):	# Skip header lines
						lc[np.argmax(lc == 0)] = float(line.split('\t')[-1])
				fluxes[x] = np.divide(lc, np.median(lc))	# Normalize around median
			progress(x+1, len(files))						# Update progress
		return files, fluxes								# Return

	# Categorize EBs through chi^2 value ceiling
	def categorize_ebs(fR, R, D, C, vc):

		# Compute chi^2 values, geometric similarity of D1 to D2 fluxes
		def compute_chis(O, M, f):
			fF = []								# Fitted EBs array
			progress(0, len(O))					# Start progress
			for x in range(len(O)):				# Loop through observed EBs
				for y in range(len(M)):			# Loop through model EBs
					chi = sum((O[x]-M[y])**2.)	# Compute chi^2 difference
					if chi <= vc:				# Compare to value ceiling
						fF.append(f[x])			# Record match
						break					# Break model EB loop upon match
				progress(x+1, len(O))			# Update progress
			fNF = [i for i in f if i not in fF]	# No fit files
			NF = [O[f.index(i)] for i in fNF]	# No fit fluxes
			return fF, fNF, NF 					# Return matched EBs

		print '\n\tCategorizing EBs: Detached test w/ VC = %s...' %str(vc)
		fRD, fRND, RND = compute_chis(R, D, fR)		# Indetify real detached EBs
		print '\n\tCategorizing EBs: Contact test w/ VC = %s...' %str(vc)
		fRC, fRNF, RNF = compute_chis(RND, C, fRND)	# Indetify real contact EBs
		return fRD, fRC, fRNF						# Return results

	# Find optimal chi^2 value ceiling for ideal categorization
	def optimize_ceiling(fD, D, fC, C, fR, R):
		vc = np.insert(np.arange(0.05, 0.35, 0.05), 0, 0.01)

		sizes = []			# Array to record categorized distribution sizes
		files = []			# Save names for post optimization
		mp = [2003, 873] 	# Morph. parameter distribution, below and above 0.7
		for i in vc:		# Loop through value ceilings

			fRD, fRC, fRNF = categorize_ebs(fR, R, D, C, i)	# Categorize EBs with i = value ceiling
			sizes.append([len(fRD), len(fRC), len(fRNF)])	# Record distribution sizes
			files.append([fRD, fRC, fRNF])					# Record categorization results
			
		sizes = map(list, zip(*sizes))	# Transpose arrays
		files = map(list, zip(*files))	# ----------------

		pl.figure('vc', figsize = (6, 4))						# Create plot
		pl.plot(vc, sizes[0], c='r', label = 'Detached')		# Detached EBs
		pl.plot(vc, sizes[1], c='b', label = 'Contact')			# Contact EBs
		pl.plot(vc, sizes[2], c='g', label = 'Uncategorized')	# Leftover EBs
		pl.axhline(y=mp[0], xmax = 9, c='r', ls = '--', label = 'Below 0.7 Morph.')	# Presumable detached EBs
		pl.axhline(y=mp[1], xmax = 9, c='b', ls = '--', label = 'Above 0.7 Morph.')	# Presumable contact EBs
		pl.legend(loc='best'); pl.xlim([0, vc[-1]])
		pl.xlabel('Chi^2 Value Ceiling')	
		pl.ylabel('Lightcurve Count of Distribution')
		pl.savefig('plots/chi_test_opt.png')

		# Return ceiling that categorizes closest along morphology parameter distribution
		diffsum = [abs(mp[0] - i[0]) + abs(mp[1] - i[1]) + i[2] for i in map(list, zip(*sizes))]
		i = diffsum.index(min(diffsum))								# Indentify optimized value
		print '\n\n\tOptimized chi^2 value ceiling: %s' %str(vc[i])	# Display optimized value
		return files[0][i], files[1][i], files[2][i], vc[i]			# Return optimized value

	# Find best training set for categorized real EBs
	def make_training_set(O, M, f, vc):
		fT = []								# Training model EBs array
		progress(0, len(M))					# Start progress
		for x in range(len(M)):				# Loop through model EBs
			for y in range(len(O)):			# Loop through observed EBs
				chi = sum((M[x]-O[y])**2.)	# Compute chi^2 difference
				if chi <= vc:				# Compare to value ceiling
					fT.append(f[x])			# Record file name
			progress(x+1, len(M))			# Update progress
		
		# Get most common best fits, 5 training lightcurves per real EB
		fT = np.asarray(Counter(fT).most_common()).T[0][:5*len(O)]
		
		T = [M[f.index(x)] for x in fT]		# Get fluxes
		return fT, T						# Return training set

	# Reformat and save training lightcurves for EBAI
	def save_training(names, fluxes, eb_type):
		files = [x.replace('pf.'+eb_type[-3:], eb_type[-3:]) for x in names]						# Get original PHOEBE computed EB file names
		pp = 5 if 'dlc' in eb_type else 4 															# Number of principal parameters   						
		pars = np.asarray([[float(y[12:]) for y in list(islice(open(x, 'r'), pp))] for x in files]).T # Extract principal parameters 
		bounds = open('bounds.%s.txt' %eb_type[-3:], 'w')											# Create parameter boundary file
		[bounds.write('%s\t%s\n' %(str(min(x)), str(max(x)))) for x in pars] 						# Write boundaries to file            
		plot_params(pars, 'plots/', 'trainEBs.%s.%iK_exemplars.png' %(eb_type[-3:], len(files)/1e3))		# Plot parameter distributions

		print '\n\tSaving training lightcurves in ANN-ready format...'
		for x in range(len(fluxes)):
			output = open('lightcurves/%i.%s' %(x, eb_type), 'w')	# Create output lightcurve file
			[output.write('%f\n' %y) for y in fluxes[x]]			# Save fluxes		
			output.write('\n'+'\n'.join(map(str, pars.T[x])))		# Append principal parameters

	# Reformat and save real EB lightcurves for EBAI
	def save_real(files, fluxes, eb_type):
		print '\n\tSaving real EB lightcurves in ANN-ready format...'
		for x in range(len(fluxes)):
			output = open('lightcurves/%i.%s' %(x, eb_type), 'w')	# Create output lightcurve file
			[output.write('%f\n' %y) for y in fluxes[x]]			# Save fluxes
			output.write('\n%s' %files[x].split('.')[1])			# Append object ID

	# Main chi_tesst() function
	def main():

		# Read in and normalize fluxes
		print '\n\tLoading & normalizing synthetic detached EB flux data...'
		fD, D = load_lcs('../model_lightcurves/*.pf.dlc')
		print '\n\tLoading & normalizing synthetic contact EB flux data...'
		fC, C = load_lcs('../model_lightcurves/*.pf.clc')
		print '\n\tLoading & normalizing real EB flux data...'
		fR, R = load_lcs('lightcurves/*.pf.rlc')

		# Find optimal chi^2 value ceiling for ideal categorization
		fRD, fRC, fRNF, vc = optimize_ceiling(fD, D, fC, C, fR, R)

		RD = [R[fR.index(x)] for x in fRD]	# Get detached fluxes
		RC = [R[fR.index(x)] for x in fRC] 	# Get contact fluxes
		if len(fRNF) > 0:					# Print unrepresented EBs
			print '\n\tThe following %i EBs could not be categorized...' %len(fRNF)
			for x in fRNF: 
				print '\t\t%s' %x.split('/')[-1].split('.')[1]

		# Chi-test #2: find geometrically similar training sets
		print '\n\tIdentifying best training set for ANN to recognize detached EBs...'
		fTD, TD = make_training_set(RD, D, fD, vc)
		print '\n\tIdentifying best training set for ANN to recognize contact EBs...'
		fTC, TC = make_training_set(RC, C, fC, vc)
		
		# Reformat and save lightcurves for EBAI, also plots parameter distributions
		save_training(fTD, TD, 'train.dlc'); save_real(fRD, RD, 'real.dlc')
		save_training(fTC, TC, 'train.clc'); save_real(fRC, RC, 'real.clc')
		
	setup()
	main()

# Identify optimal ANN parameters
def optimize_ann():

	# Create slurm job for main
	def setup():
		if 'slurm' not in sys.argv:
			for eb_type in ['dlc', 'clc']: # Loop through EB types

				# EB type dependent EBAI arguments
				pp = 5 if 'dlc' in eb_type else 4 # Number of principal parameters
				numT = len(list(glob.iglob('lightcurves/*train.%s' %eb_type)))
				formT = '--data-format %d.train.' + eb_type
				dir = '--data-dir %s/lightcurves' %sett[0]
				bounds = '--param-bounds %s/bounds.%s.txt' %(sett[0], eb_type)
				procs = int(numT/500) + (8 - int(numT/500) % 8) # ~500 LCs / processor

				# ANN parameters to demo
				hid = range(5, 51, 5) #[x for x in range(5, 81, 5) for _ in range(3)] 
				lrp = [0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0]
						
				# Create and submit slurm job (hidden layers)
				command = ['mpirun ebai.mpi -t -i 10000 -s %i -n %i:%i:%i --lrp 0.01 %s %s %s' %(numT-1, int(sett[1])+1, x, pp, dir, formT, bounds) for x in hid]
				command.append('python ebai.py 4 slurm hid %s' %eb_type)
				run_job(command, 'jobs/ebai.4.hid.%s' %eb_type, output = ['%s/ann/%s.hid.%s.txt' %(sett[0], x, eb_type) for x in hid], processors = procs)
						
				# Create and submit slurm job (learning rate)
				command = ['mpirun ebai.mpi -t -i 100 -s %i -n %i:30:%i --lrp %f %s %s %s' %(numT-1, int(sett[1])+1, pp, x, dir, formT, bounds) for x in lrp]
				command.append('python ebai.py 4 slurm lrp %s' %eb_type)
				run_job(command, 'jobs/ebai.4.lrp.%s' %eb_type, output = ['%s/ann/%s.lrp.%s.txt' %(sett[0], x, eb_type) for x in lrp], processors = procs)
			exit() # Terminate

	# Load ANN training output file
	def load_train(files):
		cfs = [np.loadtxt(x).T[1] for x in files] 									# Get learning curve
		print files
		print cfs
		if 'lrp' in sys.argv: vals = [re.findall(r'\d+\.\d+', x) for x in files]	# Get decimals from files
		if 'hid' in sys.argv: vals = [re.findall(r'\d+', x) for x in files]			# Get integers from files
		yx = zip(vals, cfs)
		yx.sort()
		cfs = [x for y, x in yx]
		vals.sort()
		return cfs, vals				# Sort arrays and return

	# Find optimal ANN learning rate
	def optimal_lrp(cfs, vals, tag):
		
		# Plot LRPs vs. cost function
		pl.figure(1, figsize = (8, 8))
		for x in range(len(cfs)):
			pl.plot(np.linspace(1, len(cfs[0]), len(cfs[0])), cfs[x], label=str(vals[x]))
		pl.legend(loc='best')
		pl.xlabel('Training Iterations')
		pl.ylabel('Cost Function')
		pl.savefig('plots/lrp.%s.png' %tag)
		
		fcfs = [cfs[x][-1] for x in range(len(cfs))] # Get last elements of each sub-array
		
		# Display and save fastest converging LRP
		lrp = vals[fcfs.index(min(fcfs))][0]
		print '\n\tFastest converging learning rate = %s' %lrp
		print '\n\tSaving optimal learning rate to settings file...'
		if 'dlc' in tag: sett[3] = lrp # detached
		if 'clc' in tag: sett[4] = lrp # contact
		ebai_setup('w', preset = sett)

	# Find optimal ANN topology
	def optimal_top(cfs, vals, tag):

		width = 0.8 # Plot parameter
		fcfs = [cfs[x][-1] for x in range(len(cfs))] # Get last elements of each sub-array
			
		# Plot topology vs. cost function
		pl.figure(2, figsize = (8, 8))
		pl.bar(np.arange(len(cfs)), [cfs[x][-1] for x in range(len(cfs))], width)
		pl.xticks(np.arange(len(cfs)) + width/2.0, vals)
		pl.xlabel('Hidden Layers')
		pl.ylabel('Cost Function (%s training iterations)' %str(len(cfs[0])))
		pl.savefig('plots/hidden.%s.png' %tag)

		hid = 30 ## FORMULA TO GET BEST HIDDEN LAYER COUNT, TO AUTOMATE ##

		# Set and display optimal topology
		if 'dlc' in tag: 
			sett[5] = '%i:%s:5' %(int(sett[1])+1, hid)
			print '\n\tOptimal ANN topology = %s' %sett[5]
		if 'clc' in tag:
			sett[6] = '%i:%s:4' %(int(sett[1])+1, hid)
			print '\n\tOptimal ANN topology = %s' %sett[6]
		
		# Save topology setting
		print '\n\tSaving optimal topology to settings file...'
		ebai_setup('w', preset = sett)
	
	# Main function
	def main():
		print '\n\tOptimizing ANN...'
		numT = len(list(glob.iglob('lightcurves/*train.%s' %sys.argv[4])))
		cfs, vals = load_train(list(glob.iglob('ann/*%s.%s' (sys.argv[3], sys.argv[4]))))
		if 'hid' in sys.argv: optimal_top(cfs, vals, '%s.%iK_exemplars' %(sys.argv[4], numT/1e3)) 	# Plot and record optimal topology figure
		if 'lrp' in sys.argv: optimal_lrp(cfs, vals, '%s.%iK_exemplars' %(sys.argv[4], numT/1e3)) 	# Plot and record optimal learning rate figure

	setup()
	main()

# Initialize ANN Training
def train_ann():

	# Create slurm job for main
	def setup():
		if 'slurm' not in sys.argv:
			for eb_type in ['dlc', 'clc']: # loop through EB types

				# EBAI arguments
				iters = 100 									# Training iterations
				dir = '--data-dir %s/lightcurves' %sett[0]
				pp = 5 if 'dlc' in eb_type else 4 				# Number of principal parameters
				numT = len(list(glob.iglob('lightcurves/*train.%s' %eb_type)))
				formT = '--data-format %d.train.' + eb_type
				bounds = '--param-bounds %s/bounds.%s.txt' %(sett[0], eb_type)
				procs = int(numT/500) + (8 - int(numT/500) % 8) # ~500 LCs / processor
				if 'dlc' in eb_type: par = [sett[3], sett[5]] 	# LRP and topology
				if 'clc' in eb_type: par = [sett[4], sett[6]] 	

				# Run training
				command = ['mpirun ebai.mpi -t -i %i -n %s -s %i --lrp %s %s %s %s' %(iters, par[1], numT-1, par[0], dir, formT, bounds)]
				command.append('python ebai.py 5 slurm %s' %eb_type)
				output = '%s/ann/train.%s.%iM_iters.%iK_exemplars.txt' %(sett[0], eb_type, iters/1e6, numT/1e3)
				run_job(command, 'jobs/ebai.5.%s' %eb_type, output, procs)
			exit()
	
	# Main train_ann() function
	def main():
		os.system('mv ../h2o.weights ann/h2o.%s.weights' %sys.argv[3]) # Move and rename weight matrix files
		os.system('mv ../i2h.weights ann/i2h.%s.weights' %sys.argv[3])

	setup()
	main()

# Analyze ANN solutions
def recognition():

	# Create slurm job for main
	def setup():
		if 'slurm' not in sys.argv:
			for eb_type in ['dlc', 'clc']: # loop through EB types

				# EBAI arguments
				iters = 3000000 # Training iterations
				dir = '--data-dir %s/lightcurves' %sett[0]
				pp = 5 if 'dlc' in eb_type else 4 # Number of principal parameters
				weights = '--i2h i2h.%s.weights --h2o h2o.%s.weights' %(eb_type, eb_type)
				numT = len(list(glob.iglob('lightcurves/*train.%s' %eb_type)))
				numR = len(list(glob.iglob('lightcurves/*real.%s' %eb_type)))
				formT, formR = '--data-format %d.train.' + eb_type, '--data-format %d.real.' + eb_type
				bounds = '--param-bounds %s/bounds.%s.txt' %(sett[0], eb_type)
				if 'dlc' in eb_type: par = [sett[3], sett[5]] # LRP and topology
				if 'clc' in eb_type: par = [sett[4], sett[6]] 	

				# Command and output - foward propagation of model and real EBs
				command = ['ebai -r -n %s --lrp %s %s %s -s %i %s %s' %(par[1], par[0], bounds, weights, numT, dir, formT)]
				command.append('ebai -r -n %s --lrp %s %s %s -s %i %s %s --unknown-data' %(par[1], par[0], bounds, weights, numR, dir, formR))
				output = ['%s/ann/recog.trainEBs.%s.%iM_iters.%iK_exemplars.txt' %(sett[0], eb_type, iters/1e6, numT/1e3)]
				output.append('%s/ann/recog.realEBs.%s.%iM_iters.%iK_exemplars.txt' %(sett[0], eb_type, iters/1e6, numT/1e3))

				command.append('python ebai.py 6 slurm %s' %eb_type) # Post-processing command
				run_job(command, 'jobs/ebai.6.%s' %(mode, eb_type))  # Create and submit slurm job
			exit()

	# Unnormalize parameter solutions based on boundaries
	def unnormalize(pars):
		pars = [pars[i] for i in range(len(pars)) if i%2 == 0]
		bounds = np.loadtxt('lightcurves/bounds.%s.txt' %eb_type)
		return [bounds[x][0] + (pars[x]-0.1)/(0.9-0.1)*(bounds[x][1]-bounds[x][0]) for x in range(len(pars))]
	
	# Plot distribution
	def plot_2dist(pars, npars, label, tag):
		save_par = ['alpha.', 'beta.', 'gamma.', 'delta.', 'epsilon.']
		label = [[r'$T_2/T_1$']*2, [r'$\rho_1 + \rho_2$', r'$M_2/M_1$'], 
				[r'$e sin(\omega)$', r'$(\Omega^I-\Omega)/(\Omega^I-\Omega^O)$'],
				[r'$e cos(\omega)$', r'$sin(i)$'], [r'$sin(i)$']*2]
		for x in len(pars):
			w1, w2 = np.ones_like(pars[x])/len(pars[x])*100, np.ones_like(npars[x])/len(npars[x])*100
			b1 = (max(pars[x])-min(pars[x]))/(max(max(pars[x]), max(npars[x]))-min(min(pars[x]), min(npars[x]))) * 40
			b2 = (max(npars[x])-min(npars[x]))/(max(max(pars[x]), max(npars[x]))-min(min(pars[x]), min(npars[x]))) * 40
			binwidth = (max(max(pars[x]), max(npars[x])) - min(min(pars[x]), min(npars[x])))/50.
			bin = np.arange(min(min(pars[x]), min(npars[x])), max(max(pars[x]), max(npars[x])) + binwidth, binwidth)
			f = pl.figure(save_file, figsize = (6, 5))
			h = pl.hist(pars[x], bins=bin, weights=w1, histtype='bar', align='mid', orientation='vertical', label='PHOEBE-Computed EB')
			j = pl.hist(npars[x], bins=bin, weights=w2, histtype='bar', align='mid', orientation='vertical', label='Kepler EB')	
			pl.legend(loc='best'); pl.xlabel(label[0 if 'dlc' in tag else 1][x]); pl.ylabel('Frequency [%]')
			pl.savefig('plots/' + save_par[x] + tag); pl.close(f)

	# Plot parameter mapping
	def plot_mapping(data, name):
		print '\n\tPlotting input to output parameter mapping...'
		
		# Plot values
		pl.figure(name, figsize = (6, 8))
		s1 = pl.scatter(data.T[1][data.T[0] != 0.5], data.T[0][data.T[0] != 0.5], c='r', marker='o')
		s2 = pl.scatter(data.T[3][data.T[2] != 0.5], data.T[2][data.T[2] != 0.5]+0.5, c='b', marker='o')
		s3 = pl.scatter(data.T[5][data.T[4] != 0.5], data.T[4][data.T[4] != 0.5]+1.0, c='g', marker='o')
		s4 = pl.scatter(data.T[7][data.T[6] != 0.5], data.T[6][data.T[6] != 0.5]+1.5, c='y', marker='o')
		
		# Plot features
		x = np.linspace(0, 1, 30)
		pl.plot(x, x + 0.0, 'k'); pl.plot(x, x + 0.5, 'k')
		pl.plot(x, x + 1.0, 'k'); pl.plot(x, x + 1.5, 'k')
		pl.xlabel('Input Parameter Values')
		pl.ylabel('Output Parameter Values')
		pl.xlim(0, 1)

		if 'dlc' in eb_type: # Detached plot specifics
			s5 = pl.scatter(data.T[9][data.T[8] != 0.5], data.T[8][data.T[8] != 0.5]+2.0, c='c', marker='o')
			pl.plot(x, x + 2.0, 'k') # for 5th pp
			pl.legend((s1, s2, s3, s4, s5), (lab1[:5]), loc='best')
			pl.ylim(0, 3.5)
		elif 'clc' in eb_type: # Contact plot specifics
			pl.legend((s1, s2, s3, s4), (lab4[:4]), loc='best')
			pl.ylim(0, 3.0)
		
		pl.savefig('plots/'+name) # Save plot

	# Plot learning curve
	def plot_learning(cf, lc_count, name):
		print '\n\tPlotting learning curve...'
		pl.figure('cost', figsize = (6, 4))
		pl.plot(np.log10(range(1, len(cf)+1), cf/lc_count, c='r'))
		pl.xlabel('Training Iterations (log scale)')
		pl.ylabel('Cost Function / Exemplar')
		pl.savefig('plots/'+name)

	# Save parameter solutions
	def save_solutions(npars):
		print '\n\tCreating object ID and solution estimates table...'
		file = open('solutions.%s.%s.txt' %(sett[0], eb_type), 'w')
		[np.loadtxt(f)[-1] for f in list(glob.iglob('lightcurves/*train.%s' %eb_type))]
		[file.write('%s\t' %x + '\t'.join(map(str, npars.T)) + '\n') for x in id]
		file.close()
								   
	# Main post_processing function
	def main(eb_type):
		tagT = 'trainEBs.%s.%iM_iters.%iK_exemplars.png' %(eb_type, iters/1e6, numT/1e3) 		# Synthetic EB plot tag
		tagR = 'realEBs.%s.%iM_iters.%iK_exemplars.png' %(eb_type, iters/1e6, numT/1e3) 		# Real EB plot tag

		# Import dlc files
		print '\n\tImporting files...'
		train = np.loadtxt(str(glob.iglob('ann/*train.%s' %eb_type))).T[1]
		recogT = np.loadtxt(str(glob.iglob('recog.trainEBs.%s' %eb_type)))
		recogR = np.loadtxt(str(glob.iglob('recog.realEBs.%s' %eb_type)))

		numT, numR = len(recogT.T[0]), len(recogR.T[0])					# Number of each of type of lightcurve
		npars = unnormalize(recogR.T)									# Unnormalize solution estimates
		plot_learning(train, len(recogT.T[0]), 'learning.%s' %tagT)		# Plot learning curve
		plot_mapping(recogT, 'i2o.%s' %tagT)							# Plot input to output mapping
		save_solutions(recogR)											# Save dlc parameter solutions
		pp = 5 if 'dlc' in eb_type else 4 								# Number of principal parameters

		# Get principal parameters
		print '\n\tReading in principal parameters...'
		pars = [np.loadtxt(f)[-pp:] for f in list(glob.iglob('lightcurves/*train.%s' %eb_type))]

		# Plot solved parameter distributions
		plot_params(npars, eb_type, 'plots/', tagR)

		# Plot parameter distribution overlaps
		plot_2dist(pars, npars, 'realEBs_v_trainEBs.%s.%iM_iters.png' %(eb_type, iters/1e6)) 

		## ERROR COMPUTATION & PLOTTING ##
		## 0.5 INPUT VALUE ERROR ##

		"""# Plot input distributions
		# get [-5] of LC files and look for 0.5 spike
		# issue

		# Compute errors
		print '\n\n  Computing errors between parameter distributions...'
		for a in range(len(pars)):
		error = [(pars[a][x] - gen_recog.T[2*a][x]) / pars[a][x] * 100 for x in range(len(pars[a]))]"""

	####
	setup()
	main(sys.argv[3])
####
if mode == 0: ebai_setup('w')		# Edit settings file
if mode == 1: compute_lightcurves()	# Compute theoretical EBs via PHOEBE
if mode == 2: polyfit_realEBs()		# Polyfit real EB lightcurves
if mode == 3: chi_tests()			# Categorize EBs, identify best training sets 
if mode == 4: optimize_ann()		# Optimize ANN parameters
if mode == 5: train_ann()			# Train ANNs
if mode == 6: recognition()			# Recognize EBs and analyze solutions