# Eclipsing Binaries via Artificial Intelligence - Hause, Prsa et al. 2017

# Basic math, arrays
import numpy as np

# Interaction with system
import sys
import os
import glob

# Basic plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl

# Labels for plotting
labels = [
    [r'$T_2/T_1$'] * 2,
    [r'$\rho_1 + \rho_2$', r'$M_2/M_1$'],
    [r'$e sin(\omega)$', r'$(\Omega^I-\Omega)/(\Omega^I-\Omega^O)$'],
    [r'$e cos(\omega)$', r'$sin(i)$'],
    [r'$sin(i)$'] * 2
]


# EBAI setup - settings and directories
def ebai_setup(key, preset=None):

    # Create default settings file
    def make_settings(fname,
                      values=[None, '500', None, None, None, None, None]):

        with open(fname, 'w') as file:

            working_dir = values[0] if values[0] else (
                raw_input('\n\tReal EB data set name: ')
            )

            file.write(
                '%s= Current real EB workset name of directory to work in\n'
                '%s= Data points per lightcurve\n' 
                '%s= Email for slurm jobs\n'
                '%s= Fastest converging learning rate (detached)\n'
                '%s= Fastest converging learning rate (contact)\n'
                '%s= Fastest converging ANN topology (detached)\n'
                '%s= Fastest converging ANN topology (contact)\n' % (
                    working_dir,
                    values[1],
                    values[2],
                    values[3],
                    values[4],
                    values[5],
                    values[6]
                )
            )

        return np.genfromtxt(
            fname,
            delimiter='=',
            dtype=str
        ).T

    # Main function
    def main(key, preset):

        # Reading / creating mode
        if key == 'r':

            # Check for settings file to load
            try:
                sett, des = np.genfromtxt(
                    'ebai.settings',
                    delimiter='=',
                    dtype=str
                ).T

            # Create default settings file if none found
            except:
                try:
                    sett, des = np.genfromtxt(
                        '../ebai.settings',
                        delimiter='=',
                        dtype=str
                    ).T

                except:
                    sett, des = make_settings('ebai.settings')

            # Check for / make directories
            def make_dir(dir):
                try:
                    os.mkdir(dir)
                except:
                    pass

            # Make directories if they don't exist
            if 'slurm' not in sys.argv and 'subprocess' not in sys.argv:
                make_dir('model_lightcurves/')
                make_dir('phoebe/')
                make_dir(sett[0])
                os.chdir(sett[0])

                # Subdirectories of sett[0]
                make_dir('lightcurves/')
                make_dir('jobs/')
                make_dir('plots/')
                make_dir('ann/')

            # Change working directory
            elif 'subprocess' not in sys.argv:
                os.chdir(sett[0])

        # Writing / editing mode
        if key == 'w':

            # Manually edit settings file
            if preset != None:
                sett, des = make_settings(
                    fname='../ebai.settings',
                    values=preset
                )

            # Write optimal ANN parameters to settings file
            else:
                sett, des = np.genfromtxt(
                    '../ebai.settings',
                    delimiter='=',
                    dtype=str
                ).T

                # Display settings
                for i in range(len(sett)):
                    print(
                        '\n\t({}){}\n\t    Current value: {}'.format(
                            i + 1,
                            des[i],
                            sett[i]
                        )
                    )

                # Change settings
                for x in range(len(sett)):
                    ans = input('\n\tSetting to change (0 to exit): ')

                    if ans in range(1, 7):
                        sett[ans - 1] = raw_input('\n\tInput new value: ')

                    else:
                        break

                # Save new settings
                sett, des = make_settings(
                    fname='../ebai.settings',
                    values=sett
                )

        return sett

    # Return settings
    return main(key,
                preset)


# Start-up instructions
def ebai_instructs(dir):

    if 'subprocess' not in sys.argv:
        # User prompt
        print(
            '\n [EBAI] - Eclipsing Binaries via Artificial Intelligence\n'
            '\n Positional arguments...'
            '\n (0) Edit ebai.settings file'
            '\n\t- Automatically created if not in script directory'
            '\n (1) Use PHOEBE to compute model EB lightcurves'
            '\n\t- Creates original simulation and polyfit-ed files '
            'for each lightcurve'
            '\n (2) Polyfit real EB lightcurves'
            '\n\t- Reads in real EBs from %s/lightcurves/'
            '\n\t- List of files (col. 1) and object IDs (col. 2) in %s'
            '\n\t- Lightcurve format: phases (col. 1) and fluxes (col. 2)'
            '\n (3) ANN training set optimization'
            '\n\t- Verify geometric overlap between computed and real EBs'
            '\n\t- Identify best model lightcurves for ANN training'
            '\n\t- Reformat all lightcurve files to be ANN-readable\n'
            '\n (4) ANN parameter optimization'
            '\n\t- Optimize learning rate & topology parameters\n'
            '\n (5) ANN training'
            '\n\t- Train ANN and compute solution estimates for real EBs\n'
            '\n (6) ANN recognition'
            '\n\t- Categorizes EBs by testing feasibility of ANN solutions'
            '\n\t- Solution analyses will be saved in %s/plots/\n' % (
                dir, dir, dir
            )
        )

        if len(sys.argv) == 1:
            exit()

        elif sys.argv[1] not in map(str, range(7)):
            print '\n\tERROR: Invalid argument\n'
            exit()

        else:
            print ' Mode: ' + sys.argv[1]

    return int(sys.argv[1])


sett = ebai_setup('r')
mode = ebai_instructs(sett[0])

# Zips lists and sorts together
def sorter(list_A, list_B):
    yx = zip(list_A, list_B)
    yx.sort()
    list_B = [x for y, x in yx]
    list_A.sort()
    return list_A, list_B


# Progess bar function
def progress(done, total, bar_length=25, progress=''):

    for i in range(bar_length):

        if i < int(bar_length * done / total):
            progress += '>'

        else:
            progress += ' '

    sys.stdout.write(
        '\r\t\t[%s] - %i of %i processed (%.2f%%)' % (
            progress,
            done,
            total,
            done * 100. / total
        )
    )

    sys.stdout.flush()


# Create slurm job
def submit_job(commands, job_name, processors=1):
    with open('%s.sh' % job_name, 'w') as job:

        job.write(
            '#!/bin/bash'
            '\n#SBATCH -J %s'
            '\n#SBATCH -p big'
            '\n#SBATCH -N 1'
            '\n#SBATCH -n %i'
            '\n#SBATCH -t 2-00:00:00'
            '\n#SBATCH -o %s/%s.out'
            '\n#SBATCH -D %s' % (
                job_name.split('/')[-1],
                processors,
                sett[0], job_name,
                os.getcwd().replace('/' + sett[0], '')
            )
        )

        if '@' in sett[2]:
            job.write(
                '\n#SBATCH --mail-type=BEGIN,END,FAIL'
                '\n#SBATCH --mail-user=%s\n' % sett[2]
            )

        # Write jobs to run
        for x in commands:
            if x[1]:
                job.write('\n' + x[0] + ' > ' + x[1])
            else:
                job.write('\n' + x[0])

    os.system('sbatch %s.sh' % job_name)


# Polyfit lightcurve
def polyfit(in_file, out_file, pf_order=2, iters=10000):
    command = 'polyfit -o %i -i %i -n %s -c 0 1 --find-knots --find-step ' % (
        pf_order,
        iters,
        sett[1]
    )

    os.system(
        '%s%s > %s' % (
            command,
            in_file,
            out_file
        )
    )


# Calculate effective potential (w1) of the primary or (w2) of the secondary
def potential(star, D, q, r, F, lmbda, nu):

    """ Args:
     D      .. instantaneous separation between components
               in units of semi-major axis (a)
     q      .. mass ratio (secondary over primary)
     r      .. star radius in units of semi-major axis (a)
     F      .. synchronicity parameter
     lambda .. direction cosine
     nu     .. direction cosine
    """

    def primary_pot(D, q, r, F, lmbda, nu):
        return (
            1 / r +
            q * (D**2 + r**2 - 2 * r * lmbda * D)**(-1 / 2) -
            r * lmbda / D**2 -
            (1 / 2) * (F**2) * (1 + q) * (r**2) * (1 - nu**2)
        )

    if star is 1:
        return primary_pot(D, q, r, F, lmbda, nu)

    if star is 2:
        return (
            q * primary_pot(D, 1. / q, r, F, lmbda, nu) +
            (1 / 2) * (q - 1) * q
        )

# Calculate critical effective potentials of both stars (through L1 and L2)
def critical_pot(q, F, e):

    D, xL, dxL = 1 - e, 0.5, 1.1e-6

    while abs(dxL) > 1e-6:
        xL = xL + dxL

        Force = (
            F**2 * (q + 1) * xL -
            1 / xL**2 -
            q * (xL - D) / abs((D - xL)**3) -
            q / D**2
        )

        dxLdF = 1 / (
            F**2 * (q + 1) +
            2 / xL**3 +
            2 * q / abs((D - xL)**3)
        )

        dxL = -1 * Force * dxLdF

    L1crit = (
        1 / xL +
        q * (D**2 + xL**2 - 2 * xL * D)**(-1 / 2) -
        xL / D**2 +
        (1 / 2) * (F**2) * (1 + q) * (xL**2)
    )

    if q > 1:
        q2 = 1 / q

    else:
        q2 = q

    D, F, dxL = 1, 1, 1.1e-6
    factor = (q2 / 3 / (q2 + 1))**(1 / 3)

    xL = (
        1.0 +
        factor +
        (1 / 3) * factor**2 +
        (1 / 9) * factor**3
    )

    while abs(dxL) > 1e-6:
        xL = xL + dxL

        Force = (
            F**2 * (q2 + 1) * xL -
            1 / xL**2 -
            q2 * (xL - D) / abs((D - xL)**3) -
            q2 / D**2
        )

        dxLdF = 1 / (
            F**2 * (q2 + 1) +
            2 / xL**3 +
            2 * q2 / abs((D - xL)**3)
        )

        dxL = -1 * Force * dxLdF

    if q > 1:
        xL = D - xL

    L2crit = (
        1 / abs(xL) +
        q * (1 / abs(xL - 1) - xL) +
        (1 / 2) * (q + 1) * xL**2
    )

    return L1crit, L2crit

# Determine feasibility of system defined by 'pars'
def feasibility(pars, eb_type):

    # Get ra surface brightness parameters
    def sb_pars():
        import phoebeBackend as pb               # PHOEBE, EB modeling engine
        from numpy.random import uniform as rng  # Monte-Carlo par. generation

        if 'dlc' in eb_type:

            """ rho1:
            obtained by sampling along a [0.025,beta-0.025] interval. The
            interval assures that the size of either star is not smaller than
            0.025 of the semimajor axis.
            """

            r2r1 = (
                1.0 +
                0.25 *
                np.sqrt(-2 * np.log(rng(0, 1))) *
                np.cos(2 * np.pi * rng(0, 1))
            )

            rho1 = pars[1] / (1 + r2r1)
            rho2 = pars[1] * r2r1 / (1 + r2r1)

            """ T1:
            the selection for T1 would most easily be  done by simply assuming
            some fixed value for T1, i.e. 6000K or such. However, since stars
            are not perfect black bodies, parameter alpha is not really an
            ideal measure of the SBR. We will thus resort to random sampling
            one more time, choosing a surface temperature on a [5000K, 30000K]
            interval. This way gravity darkening, limb darkening, reflection
            and other secular effects will introduce a systematic scatter that
            will actually be used to assess the expected 'bin' width of this
            crude 5-parameter model. The 'bin' in this context corresponds to
            the span of actual physical parameters that yield the same values
            of canonical parameters alpha thru epsilon.
            """

            T2 = 0

            while T2 < 3500:
                T1 = 5000 + rng(0, 25000)
                T2 = T1 * pars[0]

            return rho1, rho2, T1, T2

        elif 'clc' in eb_type:
            T1 = 0

            while T1 < 3500:
                T2 = 3500 + rng(0, 3500)
                T1 = T2 / pars[0]

            return (
                pb.getpar('phoebe_radius1') / 10.,
                pb.getpar('phoebe_radius2') / 10.,
                T1,
                T2
            )

    # Verify parameters feasibility
    def feasible(sbpars):
        if 'dlc' in eb_type:

            """ Determine feasibility
             Test 1: Is T2/T1 less than 0.2?
             Test 2: Is eccentricity more than 0.8?
             Test 3: (1.5 is arbitrary) pars[1] sum of radii
             Test 4: Are periastron potentials overflowing the lobe?
            """

            # Calculate potentials
            potC = critical_pot(
                q=1,
                F=1,
                e=pars[5]
            )

            pot1 = potential(
                star=1,
                D=1 - pars[5],
                q=1,
                r=sbpars[0],
                F=1,
                lmbda=1,
                nu=0
            )

            pot2 = potential(
                star=2,
                D=1 - pars[5],
                q=1,
                r=sbpars[1],
                F=1,
                lmbda=1,
                nu=0
            )

            return False if (
                pars[0] < 0.2 or
                pars[5] > 0.8 or
                1.5 * pars[1] > 1 - pars[5] or
                pot1 < potC[0] or
                pot2 < potC[0]
            ) else True

        if 'clc' in eb_type:

            """ Determine feasibility
             Test 1: --
             Test 2: --
            """

            if pars[1] < 0.15 or pars[1] > 1 / 0.15:
                return False

            elif np.arcsin(pars[3]) < np.arccos(sbpars[0] + sbpars[1]):
                return False

            else:
                return True

    # Main function
    def main():
        sbpars = sb_pars()      # Compute surface brightness parameters
        x = feasible(sbpars)    # Determine feasibility of system
        return sbpars, x

    return main()


# Plot parameter distributions
def plot_parameters(pars, save_dir, save_format):
    print '\n\tPlotting parameter distributions...'
    save_par = ['alpha.',
                'beta.',
                'gamma.',
                'delta.',
                'epsilon.']

    # Index to use
    i = 0 if 'dlc' in save_format else 1

    for x in range(len(pars)):
        if nan in pars[x]:
            print pars[x]
            print x

    for x in range(len(pars)):
        weight = np.ones_like(pars[x]) / len(pars[x])

        # Creat histogram
        fig = pl.figure(save_par[x],
                        figsize=(6, 4))

        pl.hist(
            pars[x],
            bins=50,
            weights=weight * 100,
            histtype='bar',
            align='mid',
            orientation='vertical',
            label=labels[x][i]
        )

        pl.legend(loc='best')
        pl.xlabel('Value')
        pl.ylabel('Frequency (%)')
        pl.xlim(min(pars[x]),
                max(pars[x]))
        pl.savefig(save_dir + save_par[x] + save_format)
        pl.close(fig)


# Compute and save synthetic LCs
def compute_lightcurves():

    # Create slurm job for main
    def setup():
        if 'slurm' not in sys.argv and 'subprocess' not in sys.argv:

            for eb_type in ['dlc', 'clc']:

                # Ask for number of LCs to compute
                eb = 'detached' if 'dlc' in eb_type else 'contact'
                num = input(
                    '\n\tNumber of %s EBs to compute (0 for none): ' % eb
                )

                if num:
                    num_jobs = input(
                        '\tNumber of simultaneous %s-computing jobs: ' % eb
                    )

                else:
                    continue

                # Create and submit 'jobs' slurm jobs
                step = int(num / num_jobs)

                for job_num in range(num_jobs):
                    submit_job(
                        commands=[['python ebai.py 1 slurm %s %i %i' % (
                            eb_type,
                            step * job_num,
                            step * (job_num + 1)
                        ), None]],
                        job_name='jobs/ebai.1.%s.%i' % (eb_type, job_num)
                    )

            exit()

    # Compute lightcurve
    def create_lightcurve(lc_name, eb_type):
        import phoebeBackend as pb               # PHOEBE, EB modeling engine
        from numpy.random import uniform as rng  # Monte-Carlo par. generation

        # Get random principal parameters
        def principal_params():
            if 'dlc' in eb_type:

                # Alpha = T2/T1: roughly the surface brightness ratio
                alpha = (
                    1.0 -
                    abs(
                        0.18 *
                        np.sqrt(-2 * np.log(rng(0, 1))) *
                                np.cos(2 * np.pi * rng(0, 1))
                    )
                )

                # Beta = rho1 + rho2: Fractional radii sum of both stars
                beta = 0.05 + rng(0, 0.45)

                # Attenuation (dependence of eccentricity on the sum of radii)
                e0max = 0.5 * np.exp(-6 * (beta - 0.05))

                ecc = e0max * (-1 / 3) * np.log(rng(0, 1))

                omega = rng(0, 2 * np.pi)
                gamma = ecc * np.sin(omega)
                delta = ecc * np.cos(omega)

                i_eclipse = np.arcsin(np.sqrt(1 - (0.9 * beta)**2))

                # Inclination
                incl = i_eclipse + rng(0.0, np.pi / 2 - i_eclipse)

                # Epsilon principal param.
                epsilon = np.sin(incl)

                """ IMPORTANT:
                Because of the numeric instability of Omega(q) for small q,
                the roles of stars are changed here: q > 1, star 1 is smaller
                and cooler, star 2 is hotter and larger.
                """

                return [alpha,
                        beta,
                        gamma,
                        delta,
                        epsilon,
                        ecc,
                        omega,
                        incl]

            elif 'clc' in eb_type:

                # Alpha = T2/T1: roughly the surface brightness ratio
                alpha = 1. / (
                    1 -
                    abs(
                        0.14 *
                        np.sqrt(-2 * np.log(rng(0, 1))) *
                        np.cos(2 * np.pi * rng(0, 1))
                    )
                )

                # Beta = mass ratio:
                beta = 1. / (
                    1 -
                    0.22 *
                    abs(
                        np.sqrt(-2 * np.log(rng(0, 1))) *
                        np.cos(2 * np.pi * rng(0, 1))
                    )
                )

                # Potentials
                potL = critical_pot(
                    q=beta,
                    F=1.,
                    e=0.
                )

                pot = potL[1] + rng(0, potL[0] - potL[1])

                """ Gamma principal parameter
                fillout factor = (Omega(L1)-Omega)/(Omega(L1)-Omega(L2))
                """
                gamma = (potL[0] - pot) / (potL[0] - potL[1])

                delta = 0.2 + rng(0, 0.8)   # Delta principal param. = sin(i)
                ecc = 0.0                   # Eccentricity
                omega = np.pi / 2.          # Argument of periastron

                return [alpha,
                        beta,
                        gamma,
                        delta,
                        pot,
                        ecc,
                        omega]

        # Calculate phase of conjunction
        def conjunction_phase(ecc, omega):

            ups_c = np.pi / 2 - omega

            E_c = 2 * np.arctan(np.sqrt((1 - ecc) / (1 + ecc)) *
                                np.tan(ups_c / 2))

            M_c = E_c - ecc * np.sin(E_c)

            return (M_c + omega) / 2 / np.pi - 0.25

        # Set default model parameters
        def default_pars(pars, sbpars):

            if 'dlc' in eb_type:
                pb.setpar('phoebe_ecc', pars[5])
                pb.setpar('phoebe_perr0', pars[6])
                pb.setpar('phoebe_teff1', sbpars[2])
                pb.setpar('phoebe_teff2', sbpars[3])

                pb.setpar(
                    'phoebe_incl',
                    180 * np.arcsin(pars[4]) / np.pi
                )
                pb.setpar(
                    'phoebe_pshift',
                    -1 * conjunction_phase(pars[5], pars[6])
                )
                pb.setpar(
                    'phoebe_lc_filter',
                    'Kepler:mean',
                    0
                )

                pb.setpar(
                    'phoebe_pot1',
                    potential(star=1,
                              D=1 - pars[5],
                              q=1,
                              r=sbpars[0],
                              F=1,
                              lmbda=1,
                              nu=0)
                )
                pb.setpar(
                    'phoebe_pot2',
                    potential(star=2,
                              D=1 - pars[5],
                              q=1,
                              r=sbpars[1],
                              F=1,
                              lmbda=1,
                              nu=0)
                )

            elif 'clc' in eb_type:
                pb.setpar('phoebe_rm', pars[1])
                pb.setpar('phoebe_pot1', pars[4])
                pb.setpar('phoebe_pot2', pars[4])
                pb.setpar('phoebe_teff1', sbpars[2])
                pb.setpar('phoebe_teff2', sbpars[3])
                pb.setpar('phoebe_pshift', 0.5)

                pb.setpar(
                    'phoebe_incl',
                    180 * np.arcsin(pars[3]) / np.pi
                )
                pb.setpar(
                    'phoebe_lc_filter',
                    'Kepler:mean',
                    0
                )

        # Set value of gravity darkening based on temperature
        def grav_darkening(sbpars):

            # Change discretely at T=lim
            lim = 7500 if 'dlc' in eb_type else 7000

            # Convective envelopes
            if sbpars[2] < lim:
                pb.setpar('phoebe_grb1', 0.32)
            if sbpars[3] < lim:
                pb.setpar('phoebe_grb2', 0.32)

            # Radiative envelopes
            if sbpars[2] >= lim:
                pb.setpar('phoebe_grb1', 1.00)
            if sbpars[3] >= lim:
                pb.setpar('phoebe_grb2', 1.00)

        # Startup phoebe
        pb.init()
        pb.configure()

        # Open a generic EB model
        if 'dlc' in eb_type:
            pb.open('../phoebe/detached.phoebe')
        elif 'clc' in eb_type:
            pb.open('../phoebe/contact.phoebe')

        # Phase tuple [-0.5, 0.5]
        phases = tuple(np.linspace(-0.5, 0.5, int(sett[1])).tolist())

        # Get random principal parameters
        pars = principal_params()

        # Surface brightness pars and determine feasibility
        sbpars, feasible = feasibility(pars, eb_type)

        if feasible:                    # Check parameter feasibility
            default_pars(pars, sbpars)  # Load default parameters
            pb.updateLD()               # Limb darkening, Van Hamme (1993)
            grav_darkening(sbpars)      # Gravity darkening
            flux = pb.lc(phases, 0)     # Get flux

            # Check for PHOEBE errors
            if flux and flux[0] and True not in np.isnan(flux):

                # Check if luminosities are outside the expected interval:
                if 'dlc' in eb_type:
                    sbr1 = pb.getpar('phoebe_sbr1')  # Primary luminosity
                    sbr2 = pb.getpar('phoebe_sbr2')  # Secondary luminosity

                    # Freeze subprocess on rejection, wait to be killed
                    if sbr2 / sbr1 < 0.1 and sbr2 > sbr1:
                        time.sleep(60)

                # Create LC file
                fname = '../model_lightcurves/%i.' % lc_name + eb_type
                fileout = open(fname, 'w')

                # Create file header
                fileout.write(
                    '# alpha   = %f\n'
                    '# beta    = %f\n'
                    '# gamma   = %f\n'
                    '# delta   = %f\n' % (
                        pars[0],
                        pars[1],
                        pars[2],
                        pars[3]
                    )
                )

                if 'dlc' in eb_type:
                    fileout.write('# epsilon = %f\n' % pars[4])
                elif 'clc' in eb_type:
                    fileout.write('# pot     = %f\n' % pars[4])

                fileout.write(
                    '# ecc     = %f\n'
                    '# omega   = %f\n'
                    '# rho1    = %f\n'
                    '# rho2    = %f\n'
                    '# Teff1   = %f\n'
                    '# Teff2   = %f\n' % (
                        pars[5],
                        pars[6] * 180 / np.pi,
                        sbpars[0],
                        sbpars[1],
                        sbpars[2],
                        sbpars[3]
                    )
                )

                if 'dlc' in eb_type:
                    fileout.write(
                        '# incl.   = %f\n'
                        '# sbr1    = %f\n'
                        '# sbr2    = %f\n' % (
                            180 * np.arcsin(pars[4]) / np.pi,
                            sbr1,
                            sbr2
                        )
                    )

                else:
                    fileout.write(
                        '# incl.   = %f\n'
                        '# sbr1    = N/A\n'
                        '# sbr2    = N/A\n' % (180 * np.arcsin(pars[3]) / np.pi)
                    )

                # Compile data points array
                data = [
                    4 * np.pi * i /
                    (pb.getpar('phoebe_plum1') + pb.getpar('phoebe_plum2'))
                    for i in flux
                ]

                # Write phases and fluxes to file
                for i in range(int(sett[1])):
                    fileout.write('%s\t%s\n' % (phases[i], str(data[i])))

                fileout.close()  # Close out lightcurve file
                pb.quit()        # Close out PHOEBE

                # Polyfit new LC file
                polyfit(
                    in_file=fname,
                    out_file='../model_lightcurves/%i.pf.' % lc_name + eb_type
                )

            # Freeze subprocess on failure, wait to be killed
            else:
                time.sleep(60)

    # Main function
    def main():
        import subprocess
        import time
        from shlex import split as splitsh

        # Initiate subprocess - subprocessing protects against PHOEBE freezing
        if 'subprocess' in sys.argv:
            create_lightcurve(
                lc_name=int(sys.argv[4]),
                eb_type=sys.argv[3]
            )
            exit()

        # Define start and end points
        print '\n\tComputing and saving theoretical lightcurves...'

        # Initialize counter
        num = int(sys.argv[4])

        # Initialize progress bar
        progress(0, int(sys.argv[5]) - int(sys.argv[4]))

        # Create subprocess for each LC - protects against PHOEBE freezing
        while num <= int(sys.argv[5]):

            # Check for successful lightcurve calculation
            if os.path.isfile('../model_lightcurves/%i.pf.%s' % (
                num,
                sys.argv[3]
            )):
                num += 1
                progress(
                    num - int(sys.argv[4]),
                    int(sys.argv[5]) - int(sys.argv[4])
                )

            # Compute lightcurve through subprocess
            else:
                command = 'python ../%s %s subprocess %s %i' % (
                    sys.argv[0],
                    sys.argv[1],
                    sys.argv[3],
                    num
                )
                proc = subprocess.Popen(splitsh(command))

                # 20 sec wait (< 20 second timeout can cause errors)
                time.sleep(20)

                # Check on process (None = ongoing, 0 = done)
                if proc.poll() == None:
                    proc.kill()

    setup()
    main()


# Polyfit real EBs
def polyfit_ebs():

    # Create slurm job for main
    def setup():
        if 'slurm' not in sys.argv:

            # Ask for user input
            num_jobs = input(
                '\n\tNumber of simultaneous polyfit jobs: '
            )
            list = raw_input(
                '\tList of file names and object IDs in %s/: ' % sett[0]
            )

            # Check to see if list exists
            try:
                np.loadtxt(list, dtype=str)

            except:
                print '\n\tERROR: \'%s\' does not exist\n' % list
                exit()

            # Create and submit njobs slurm jobs
            for job_num in range(num_jobs):
                submit_job(
                    commands='python ebai.py 2 slurm %i %i %s' % (
                        num_jobs,
                        job_num,
                        list
                    ),
                    jobs_name='jobs/ebai.2.%i' % job_num
                )

            exit()

    # Main function
    def main():

        # Get file names & object IDs
        names, ids = np.loadtxt(sys.argv[5], dtype=str).T
        files = np.split(names, int(sys.argv[3]))[int(sys.argv[4])]
        names = names.tolist()
        ids = ids.tolist()

        # Initialize progress bar
        progress(0, len(files))

        # Polyfit files
        for i in range(len(files)):
            polyfit(
                in_file='lightcurves/' + files[x],
                out_file='lightcurves/%i.%s.pf.rlc' % (
                    names.index(files[i]),
                    ids[names.index(files[i])]
                )
            )

            # Update progress
            progress(i + 1, len(files))

    setup()
    main()


# ANN training set optimization
def chi_tests():

    # Create slurm job for main
    def setup():
        if 'slurm' not in sys.argv:
            for eb_type in ['dlc', 'clc']:

                # Submit slurm job
                submit_job(
                    commands=[[
                        'python ebai.py 3 slurm chi %s' % eb_type,
                        None
                    ]],
                    job_name='jobs/ebai.3.%s' % eb_type
                )
        
            exit()

    # Load lightcurve fluxes from files of format
    def load_lcs(format):

        # Get files of format
        files = list(glob.iglob(format))

        # Start progress
        progress(0, len(files))

        # Fluxes array
        fluxes = np.zeros([len(files), int(sett[1]) + 1])

        for x in range(len(files)):

            # Open lightcurve file
            with open(files[x]) as f:
                lc = np.zeros(int(sett[1]) + 1)
                for line in f:

                    # Skip header lines
                    if not line.strip().startswith("#"):
                        lc[np.argmax(lc == 0)] = float(line.split('\t')[-1])

                # Normalize around median
                fluxes[x] = np.divide(lc, np.median(lc))

            # Update progress
            progress(x + 1, len(files))

        return files, fluxes

    # Get pblum and l3 to match M amplitude to O via least squares
    def amp_correct(obs, model):
        pts = len(obs)
        sumO, sumMO = np.sum(obs), np.sum(model * obs)
        sumM, sumMM = np.sum(model), np.sum(model * model)
        pblum = (-pts * sumMO + sumM * sumO) / (sumM**2 - pts * sumMM)
        l3 = (1. / pts) * (sumO - pblum * sumM)
        return (pblum, l3)

    # Pre-categorize EBs by verifying geometric similarity to model EBs
    def filter_ebs(obs_files, obs_lcs, model_lcs, constraint):
        print (
            '\n\tChi^2 test #1: Filtering out EBs '
            'without sufficient geometric overlap ...'
        )

        chis = [0] * len(obs_lcs)       # Lowest chi^2 value per d1 EB
        progress(0, len(obs_lcs))       # Start progress
        for x in range(len(obs_lcs)):   # Loop through obs EBs
            for model_lc in model_lcs:  # Loop through model EBs

                # Least squares amplitude fix
                a, b = amp_correct(
                    obs_lcs[x],
                    model_lc
                )

                # Apply least squares fix
                model_lc = a * model_lc + b

                # Total up chi^2 values for each comparison of obs. LC
                chis[x] += np.sum((obs_lcs[x] - model_lc)**2)

            # Get avergae chi^2 value for obs. LC
            chis[x] = chis[x] / len(model_lcs)

            # Update progress
            progress(x + 1, len(obs_lcs))

        # Sort according to chis list
        return sorter(chis, obs_lcs)[-1][:constraint]

    # Find best training set for the categorized obs EBs
    def optimize_training_set(model_files, model_lcs, obs_lcs, constraint):
        print (
            '\n\tChi^2 test #2: Determining best training '
            'set for ANN to recognize filtered EBs...'
        )

        chis = [0] * len(model_lcs)      # Lowest chi^2 value per d1 EB
        progress(0, len(model_lcs))      # Start progress
        for x in range(len(model_lcs)):  # Loop through obs EBs
            for obs_lc in obs_lcs:       # Loop through model EBs

                # Least squares amplitude fix
                a, b = amp_correct(
                    obs_lc,
                    model_lcs[x]
                )

                # Apply least squares fix
                model_lc = a * model_lcs[x] + b

                # Total up chi^2 values for each comparison of model LC
                chis[x] += np.sum((obs_lc - model_lc)**2)

            # Get avergae chi^2 value for model LC
            chis[x] = chis[x] / len(obs_lcs)

            # Update progress
            progress(x + 1, len(model_lcs))

        # Sort according to chis list
        model_lcs = sorter(chis, model_lcs)[-1][:constraint*5]
        model_files = sorter(chis, model_files)[-1][:constraint*5]

        return model_files, model_lcs

    # Reformat and save training lightcurves for EBAI
    def save_training(files, lcs, eb_type, save_format):

        # For reading parameters from files
        from itertools import islice

        # Get original PHOEBE computed EB file names
        files = [
            x.replace('pf.' + eb_type, eb_type) for x in files
        ]

        # Number of principal parameters
        pars_count = 5 if 'dlc' in eb_type else 4

        # Extract principal parameters
        pars = np.asarray([
            [float(y[12:]) for y in list(islice(open(x, 'r'), pars_count))]
            for x in files]).T

        # Create parameter boundary file and write to it
        bounds = open('ann/bounds.%s.txt' % eb_type, 'w')
        for x in pars:
            bounds.write('%f\t%f\n' % (min(x), max(x)))

        # Plot parameter distributions
        plot_parameters(
            pars=pars,
            save_dir='plots/',
            save_format='training.%s.png' % eb_type
        )

        # Create and save lightcurve files
        print '\n\tSaving training lightcurves in ANN-ready format...'

        for x in range(len(lcs)):
            with open('lightcurves/%i.%s' % (x, save_format), 'w') as f:

                # Save lightcurve flux data
                [f.write('%f\n' % y) for y in lcs[x]]

                # Append principal parameters
                f.write('\n' + '\n'.join(map(str, pars.T[x])))

    # Reformat and save real EB lightcurves for EBAI
    def save_real(files, lcs, save_format):
        print '\n\tSaving EB lightcurves in ANN-ready format...'

        for x in range(len(lcs)):
            with open('lightcurves/%i.%s' % (x, save_format), 'w') as f:

                # Save lightcurve flux data
                [f.write('%f\n' % y) for y in lcs[x]]

                # Append object ID
                f.write('\n%s' % files[x].split('.')[1])

    # Main function
    def main(eb_type):

        if 'clc' in eb_type:
            constraint = 873   # EBs above 0.7 Morph.
        if 'dlc' in eb_type:
            constraint = 2003  # EBs below 0.7 Morph.

        # Read in and normalize fluxes
        print '\n\tReading in & normalizing synthetic EBs...'
        model_files, model_lcs = load_lcs(
            '../model_lightcurves/*.pf.%s' % eb_type
        )

        print '\n\tReading in & normalizing real EB flux data...'
        obs_files, obs_lcs = load_lcs(
            'lightcurves/*.pf.rlc'
        )

        if 'dlc' in eb_type:
            # Reformat and save EB lightcurves for ANN
            save_real(
                files=obs_files,
                lcs=obs_lcs,
                save_format=sett[0] + '.lc'
            )

        # Return EBs with geometric similarity to model set
        filtered_lcs = filter_ebs(
            obs_files,
            obs_lcs,
            model_lcs,
            constraint
        )

        # Optimize training set for ANN
        model_files, model_lcs = optimize_training_set(
            model_files,
            model_lcs,
            filtered_lcs,
            constraint
        )

        # Save training set
        save_training(
            files=model_files,
            lcs=model_lcs,
            eb_type=eb_type,
            save_format='training.%s' % eb_type
        )

    setup()
    main(sys.argv[-1])

# ANN parameter optimization
def optimize_ann():

    # Create slurm job for main
    def setup():
        if 'slurm' not in sys.argv:

            # Directory of input data
            dir = '--data-dir %s/lightcurves' % sett[0]

            for eb_type in ['dlc', 'clc']:  # Loop through EB types

                jobs = []

                # Number of EBs
                num_real = len(list(glob.iglob(
                    'lightcurves/*%s.lc' % sett[0]
                )))
                num_training = len(list(glob.iglob(
                    'lightcurves/*training.%s' % eb_type
                )))

                # Number of principal parameters
                prin_pars = 5 if 'dlc' in eb_type else 4

                # Get ~500 LCs / processor
                procs = round((1.*num_training/500)%500)

                # Formats of lightcurve files
                format_training = '--data-format %d.training.' + eb_type
                format_real = '--data-format %d.' + sett[0] + '.lc'

                bounds = '--param-bounds %s/ann/bounds.%s.txt' % (
                    sett[0],
                    eb_type
                )

                # ANN trial runs - test range of ANN parameters
                for lrp in [0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0]:
                    jobs.append([
                        'mpirun ebai.mpi -t -i 1000 '
                        '-s %i -n %i:30:%i --lrp %f %s %s %s' % (
                            num_training - 1,
                            int(sett[1]) + 1,
                            prin_pars,
                            lrp,
                            dir,
                            format_training,
                            bounds
                        ),
                        '%s/ann/%s.lrp.%s.txt' % (
                            sett[0],
                            lrp,
                            eb_type
                        )
                    ])

                """
                for hidden_layers in range(5, 51, 5):
                    jobs.append([
                        'mpirun ebai.mpi -t -i 10000 '
                        '-s %i -n %i:%i:%i --lrp 0.01 %s %s %s' % (
                            num_training - 1,
                            int(sett[1]) + 1,
                            hidden_layers,
                            prin_pars,
                            dir,
                            format_training,
                            bounds
                        ),
                        '%s/ann/%s.hid.%s.txt' % (
                            sett[0],
                            hidden_layers,
                            eb_type
                        )
                    ])
                """
                # ANN parameter optimization
                jobs.append([
                    'python ebai.py 4 slurm lrp %s %i' % (eb_type, num_training),
                    None
                ])
                """
                jobs.append([
                    'python ebai.py 4 slurm hid %s %i' % (eb_type, num_training),
                    None
                ])
                """

                # Remove weight matrix files
                jobs.append(['rm *.weights', None])

                # Submit slurm job
                submit_job(
                    commands=jobs,
                    job_name='jobs/ebai.4.%s' % eb_type,
                    processors=procs
                )

            exit()

    # Load ANN training output file
    def load_train(files):
        import re

        # Get parameter decimals from file names
        if 'lrp' in sys.argv:
            vals = [float(re.findall(r'\d+\.\d+', x)[0]) for x in files]

        # Get parameter integers from file names
        if 'hid' in sys.argv:
            vals = [int(re.findall(r'\d+', x)[0]) for x in files]

        # Get learning curve
        cfs = [np.loadtxt(x).T[1] for x in files]

        # Sort arrays
        vals, cfs = sorter(vals, cfs)

        return cfs, vals

    # Find optimal ANN learning rate
    def optimal_lrp(cfs, vals, tag):

        # Plot LRPs vs. cost function
        pl.figure(1, figsize=(8, 8))
        for x in range(len(cfs)):
            pl.plot(
                np.linspace(1, len(cfs[0]), len(cfs[0])),
                cfs[x],
                label=str(vals[x])
            )

        pl.legend(loc='best')
        pl.xlabel('Training Iterations')
        pl.ylabel('Cost Function')
        pl.savefig('plots/lrp.%s.png' % tag)

        # Get last elements of each sub-array
        fcfs = [x[-1] for x in cfs]

        # Display and save fastest converging LRP
        lrp = vals[fcfs.index(min(fcfs))]

        print '\n\tFastest converging learning rate = %s' % lrp
        print '\n\tSaving optimal learning rate to settings file...'

        if 'dlc' in tag:
            sett[3] = str(lrp)
        elif 'clc' in tag:
            sett[4] = str(lrp)

        ebai_setup('w', preset=sett)

    # Find optimal ANN topology
    def optimal_top(cfs, vals, tag):

        # Plot parameter
        width = 0.8

        # Get last elements of each sub-array
        fcfs = [cfs[x][-1] for x in range(len(cfs))]

        # Plot topology vs. cost function
        pl.figure(2, figsize=(8, 8))
        pl.bar(
            np.arange(len(cfs)),
            [cfs[x][-1] for x in range(len(cfs))],
            width
        )

        pl.xticks(np.arange(len(cfs)) + width / 2.0, vals)
        pl.xlabel('Hidden Layers')
        pl.ylabel('Cost Function (%s training iterations)' % str(len(cfs[0])))
        pl.savefig('plots/hidden.%s.png' % tag)

        ## FORMULA TO GET BEST HIDDEN LAYER COUNT, TO AUTOMATE ##
        hid = 30

        # Set and display optimal topology
        if 'dlc' in tag:
            sett[5] = '%i:%s:5' % (int(sett[1]) + 1, hid)
            print '\n\tOptimal ANN topology = %s' % sett[5]
        elif 'clc' in tag:
            sett[6] = '%i:%s:4' % (int(sett[1]) + 1, hid)
            print '\n\tOptimal ANN topology = %s' % sett[6]

        # Save topology setting
        print '\n\tSaving optimal topology to settings file...'
        ebai_setup('w', preset=sett)

    # Main function
    def main():
        print '\n\tOptimizing ANN parameters...'

        cfs, vals = load_train(
            list(glob.iglob(
                'ann/*%s.%s*' % (sys.argv[3], sys.argv[4])
            ))
        )

        # Plot / record optimal learning rate figure
        if 'lrp' in sys.argv:
            optimal_lrp(
                cfs,
                vals,
                '%s.%iK_exemplars' % (sys.argv[4], int(sys.argv[5]) / 1e3)
            )

        else:
            # Plot and record optimal topology figure
            optimal_top(
                cfs,
                vals,
                '%s.%iK_exemplars' % (sys.argv[4], int(sys.argv[5]) / 1e3)
            )

    setup()
    main()

# ANN training
def train_ann():

    # Create slurm job
    def setup():

        # Directory of input data
        dir = '--data-dir %s/lightcurves' % sett[0]

        iters = 2000000                 # Training iterations
        for eb_type in ['dlc', 'clc']:  # Loop through EB types

            # Number of EBs
            num_real = len(list(glob.iglob(
                'lightcurves/*%s.lc' % sett[0]
            )))
            num_training = len(list(glob.iglob(
                'lightcurves/*training.%s' % eb_type
            )))

            # Number of principal parameters
            prin_pars = 5 if 'dlc' in eb_type else 4

            # Get ~500 LCs / processor
            procs = round((1.*num_training/500)%500)
            #procs = int(numT / 500) + 8 - (int(numT / 500) % 8)

            # Learning rate and topology
            if 'dlc' in eb_type:
                lrp, topology = sett[3], sett[5]
            if 'clc' in eb_type:
                lrp, topology = sett[4], sett[6]

            # Formats of lightcurve files
            format_training = '--data-format %d.training.' + eb_type
            format_real = '--data-format %d.' + sett[0] + '.lc'

            bounds = '--param-bounds %s/ann/bounds.%s.txt' % (
                sett[0],
                eb_type
            )

            # ANN training
            jobs = [[
                'mpirun ebai.mpi -t '
                '-i %i -n %s -s %i --lrp %s %s %s %s' % (
                    iters,
                    topology,
                    num_training - 1,
                    lrp,
                    dir,
                    format_training,
                    bounds
                ),
                '%s/ann/training.%s.txt' % (sett[0], eb_type)
            ]]

            # Move and rename weight matrix files
            jobs.append([
                'mv h2o.weights %s/ann/h2o.%s.weights' % (sett[0], eb_type),
                None
            ])
            jobs.append([
                'mv i2h.weights %s/ann/i2h.%s.weights' % (sett[0], eb_type),
                None
            ])

            # Submit slurm job
            submit_job(
                commands=jobs,
                job_name='jobs/ebai.5.%s' % eb_type,
                processors=procs
            )

    setup()


# ANN recognition
def solve_ebs():

    # Create slurm job for main
    def setup():
        if 'slurm' not in sys.argv:

            # Directory of input data
            dir = '--data-dir %s/lightcurves' % sett[0]

            iters = 1000                 # Training iterations
            for eb_type in ['dlc', 'clc']:  # Loop through EB types

                # Number of EBs
                num_real = len(list(glob.iglob(
                    'lightcurves/*%s.lc' % sett[0]
                )))
                num_training = len(list(glob.iglob(
                    'lightcurves/*training.%s' % eb_type
                )))

                # Number of principal parameters
                prin_pars = 5 if 'dlc' in eb_type else 4

                # Learning rate and topology
                if 'dlc' in eb_type:
                    lrp, topology = sett[3], sett[5]
                if 'clc' in eb_type:
                    lrp, topology = sett[4], sett[6]

                weights = (
                    '--i2h %s/ann/i2h.%s.weights '
                    '--h2o %s/ann/h2o.%s.weights' % (
                        sett[0], eb_type,
                        sett[0], eb_type
                    )
                )

                # Formats of lightcurve files
                format_training = '--data-format %d.training.' + eb_type
                format_real = '--data-format %d.' + sett[0] + '.lc'

                bounds = '--param-bounds %s/ann/bounds.%s.txt' % (
                    sett[0],
                    eb_type
                )

                # ANN recognition
                jobs = [[
                    'ebai -r -n %s --lrp %s %s %s -s %i %s %s' % (
                        topology,
                        lrp,
                        bounds,
                        weights,
                        num_training,
                        dir,
                        format_training
                    ),
                    '%s/ann/recognition.model.%s.txt' % (sett[0], eb_type)
                ]]
                jobs.append([
                    'ebai -r '
                    '-n %s --lrp %s %s %s -s %i %s %s --unknown-data' % (
                        topology,
                        lrp,
                        bounds,
                        weights,
                        num_real,
                        dir,
                        format_real
                    ),
                    '%s/ann/recognition.%s.%s.txt' % (
                        sett[0],
                        sett[0],
                        eb_type
                    )
                ])

                # Solutions analysis
                jobs.append([
                    'python ebai.py 6 slurm %s %i' % (eb_type, iters),
                    None
                ])

                # Submit slurm job
                submit_job(
                    commands=jobs,
                    job_name='jobs/ebai.6.%s' % eb_type
                )

            exit()

    # Unnormalize parameter solutions based on boundaries
    def unnormalize(pars, eb_type):

        # Filter out input values from provided file
        pars = pars[0::2]

        # Load boundary file
        bounds = np.loadtxt('ann//bounds.%s.txt' % eb_type).tolist()

        # Unnormalize parameters
        new_pars = []
        for x in bounds:
            new_pars.append(
                x[0] + (pars[bounds.index(x)] - .1) / (.9 - .1) * (x[1] - x[0])
            )
        return new_pars

    #  Categorize EBs by verifying solution feasibility
    def categorize_ebs(pars, eb_type, training_iters, num_ebs):
        print '\n\tDetermining feasibility of ANN solution estimates...'
        
        import phoebeBackend as pb

        # Creat output file
        out = open('solutions.%s.txt' % eb_type, 'w')
        out.write(
            '# Solution estimates provided by the following ANN:\n'
            '#        Number of training lightcurves: %i\n'
            '#        Training iterations: %s\n' % (
                num_ebs,
                training_iters
            )
        )

        if 'dlc' in eb_type:
            out.write(
                '#        ANN topology: %s\n'
                '#        Learning rate: %s\n'
                '# Obj. ID\tT1/T2\tp1+p2\te*sin(w)\te*cos(w)\tsin(i)\n' % (
                    sett[5],
                    sett[3]
                )
            )

        elif 'clc' in eb_type:
            out.write(
                '#        ANN topology: %s\n'
                '#        Learning rate: %s\n'
                '# Obj. ID\tT1/T2\tM2/M1\tFillout\tsin(i)\n' % (
                    sett[6],
                    sett[4]
                )
            )

        # Transpose list
        pars = map(list, zip(*pars))

        # Start progress
        progress(0, len(pars))

        # List for parameters that pass feasibility testing
        passed = []

        # Check feasibility of solution estimates, categorize EBs
        for par_set in pars:

            # Startup phoebe for sb_pars()
            pb.init()
            pb.configure()

            # Open a generic EB model
            if 'clc' in eb_type:
                pb.open('../phoebe/contact.phoebe')
            elif 'dlc' in eb_type:
                pb.open('../phoebe/detached.phoebe')

                # Compute eccentricity parameter for detached feasibility tests
                e1 = (
                    par_set[2] /
                    np.sin(np.arctan(par_set[2] / par_set[2]))
                )**2
                e2 = (
                    par_set[3] /
                    np.cos(np.arctan(par_set[2] / par_set[3]))
                )**2

                par_set.append(e1 + e2)

            pass_count = 0

            # Verify feasibility of system
            while feasibility(par_set, eb_type)[1]:

                pass_count += 1

                if pass_count == 2:

                    # Get object ID
                    ID = np.loadtxt(
                        'lightcurves/%i.%s.lc' % (pars.index(par_set), sett[0])
                    )[-1]

                    # Create new entry
                    out.write(
                        str(int(ID)) + '\t' + '\t'.join(map(str, par_set)) + '\n'
                    )

                    # Save feasible parameters
                    passed.append(par_set)

                    break

            # Close phoebe
            pb.quit()

            # Update progress
            progress(
                pars.index(par_set) + 1,
                len(pars)
            )

        # Results
        if 'dlc' in eb_type:
            out.write('# Detached EBs recognized: %i' % len(passed))
        if 'clc' in eb_type:
            out.write('# Contact EBs recognized: %i' % len(passed))

        out.close()                     # Close output file
        return map(list, zip(*passed))  # Return transposed array

    # Plot distribution
    def plot_overlaps(model_pars, obs_pars, save_format):
        save_par = ['alpha.',
                    'beta.',
                    'gamma.',
                    'delta.',
                    'epsilon.']

        for x in len(model_pars):

            # Nornalize number of pins
            binwidth = (
                max(max(model_pars[x]), max(obs_pars[x])) -
                min(min(model_pars[x]), min(obs_pars[x]))
            ) / 50.

            bins = np.arange(
                min(
                    min(model_pars[x]), min(obs_pars[x])
                ),
                binwidth + max(
                    max(model_pars[x]), max(obs_pars[x])
                ),
                binwidth
            )

            # Plot histograms
            f = pl.figure(save_file, figsize=(6, 5))

            h = pl.hist(
                model_pars[x],
                bins=bins,
                weights=np.ones_like(model_pars[x]) / len(model_pars[x]) * 100,
                histtype='bar',
                align='mid',
                orientation='vertical',
                label='PHOEBE-Computed EB'
            )

            j = pl.hist(
                obs_pars[x],
                bins=bins,
                weights=np.ones_like(obs_pars[x]) / len(obs_pars[x]) * 100,
                histtype='bar',
                align='mid',
                orientation='vertical',
                label='Kepler EB'
            )

            pl.legend(loc='best')
            pl.xlabel(labels[0 if 'dlc' in save_format else 1][x])
            pl.ylabel('Frequency [%]')
            pl.savefig('plots/' + save_par[x] + save_format)
            pl.close(f)

    # Plot parameter mapping
    def plot_mapping(data, save_name):
        print '\n\tPlotting input to output parameter mapping...'

        # Plot input and output values
        pl.figure(save_name, figsize=(6, 8))

        s1 = pl.scatter(data.T[1][data.T[0] != 0.5],
                        data.T[0][data.T[0] != 0.5],
                        c='r',
                        marker='o')

        s2 = pl.scatter(data.T[3][data.T[2] != 0.5],
                        data.T[2][data.T[2] != 0.5] + 0.5,
                        c='b',
                        marker='o')

        s3 = pl.scatter(data.T[5][data.T[4] != 0.5],
                        data.T[4][data.T[4] != 0.5] + 1.0,
                        c='g',
                        marker='o')

        s4 = pl.scatter(data.T[7][data.T[6] != 0.5],
                        data.T[6][data.T[6] != 0.5] + 1.5,
                        c='y',
                        marker='o')

        x = np.linspace(0, 1, 30)

        # Plot line guides
        pl.plot(x, x + 0.0, 'k')
        pl.plot(x, x + 0.5, 'k')
        pl.plot(x, x + 1.0, 'k')
        pl.plot(x, x + 1.5, 'k')

        pl.xlabel('Input Parameter Values')
        pl.ylabel('Output Parameter Values')

        pl.xlim(0, 1)
        pl.ylim(0, 3)

        pl.legend(
            (s1, s2, s3, s4),
            (zip(*labels[1])),
            loc='best'
        )

        # Detached plot specifics
        if 'dlc' in save_name:
            s5 = pl.scatter(data.T[9][data.T[8] != 0.5],
                            data.T[8][data.T[8] != 0.5] + 2.0,
                            c='c',
                            marker='o')

            pl.plot(x, x + 2.0, 'k')    # for 5th pp

            pl.legend(
                (s1, s2, s3, s4, s5),
                (zip(*labels)[0]),
                loc='best'
            )

            pl.ylim(0, 3.5)

        # Save plot
        pl.savefig('plots/' + save_name)

    # Plot learning curve
    def plot_learning(cost_func, num_ebs, save_name):
        print '\n\tPlotting learning curve...'

        pl.figure('cost',
                  figsize=(6, 4))

        pl.plot(
            np.log10(range(1, len(cost_func) + 1)),
            cost_func / num_ebs,
            c='r'
        )

        pl.xlabel('Training Iterations (log scale)')
        pl.ylabel('Cost Function / Exemplar')
        pl.savefig('plots/' + save_name)

    # Main function
    def main(eb_type, iters):

        # Import relevent files
        print '\n\tReading in ANN output files...'

        cost_func = np.loadtxt(
            'ann/training.%s.txt' % eb_type
        ).T[1]

        model_recog = np.loadtxt(
            'ann/recognition.model.%s.txt' % eb_type
        )

        obs_recog = np.loadtxt(
            'ann/recognition.%s.%s.txt' % (sett[0], eb_type)
        )

        # Number of principal parameters
        pars_count = 5 if 'dlc' in eb_type else 4

        # Model EB plot save format
        model_save = 'model.%s.png' % eb_type

        # Obs. EB plot save format
        obs_save = '%s.%s.png' % (sett[0], eb_type)

        # Unnormalize solution estimates
        obs_pars = unnormalize(
            pars=obs_recog.T,
            eb_type=eb_type
        )

        # Verify feasibility of solution estimates
        obs_pars = categorize_ebs(
            pars=obs_pars,
            eb_type=eb_type,
            training_iters=iters,
            num_ebs=len(model_recog.T[0])
        )

        # Plot learning curve
        plot_learning(
            cost_func=cost_func,
            num_ebs=len(model_recog.T[0]),
            save_name='learning.' + model_save
        )

        # Plot input to output mapping
        plot_mapping(
            data=model_recog,
            save_name='i2o.' + model_save
        )

        # Plot solved parameter distributions
        plot_parameters(
            pars=obs_pars,
            save_dir='plots/',
            save_format=obs_save
        )

        # Get principal parameters
        print '\n\tReading in model EB principal parameters...'
        model_pars = []

        for f in list(glob.iglob('lightcurves/*training.%s' % eb_type)):
            model_pars.append(np.loadtxt(f)[-pars_count:])

        # Plot parameter distribution overlaps
        plot_overlaps(
            model_pars=model_pars,
            obs_pars=obs_pars,
            save_format='{0}_v_model.{1}.png'.format(
                sett[0],
                eb_type
            )
        )

        ## ERROR COMPUTATION & PLOTTING ##
        ## 0.5 INPUT VALUE ERROR ?? ##

        """# Plot input distributions
        # get [-5] of LC files and look for 0.5 spike
        # issue
        # Compute errors
        print '\n\n  Computing errors between parameter distributions...'
        for a in range(len(pars)):
        error = [
            (pars[a][x] - gen_recog.T[2*a][x]) / pars[a][x] * 100
            for x in range(len(pars[a]))
        ]"""

    setup()
    main(sys.argv[3], sys.argv[4])


# Edit settings file
if mode is 0:
    ebai_setup('w')

# Compute theoretical EBs via PHOEBE
if mode == 1:
    compute_lightcurves()

# Polyfit real EB lightcurves
if mode == 2:
    polyfit_ebs()

# ANN training set optimization
if mode == 3:
    chi_tests()

# ANN parameter optimization
if mode == 4:
    optimize_ann()

# ANN training
if mode == 5:
    train_ann()

# ANN recognition
if mode == 6:
    solve_ebs()

"""
a, b = [], []
did = np.loadtxt('kepler/solutions.dlc.txt').T[0]
cid = np.loadtxt('kepler/solutions.clc.txt').T[0]
for x in did:
    if x in cid:
        b.append(x)
    else:
        a.append(x)
for x in cid:
    if x not in did:
        a.append(x)
print 'unique solutions:', len(a)
print 'uncategorize EBs:', 2876-len(a)
print 'in both:', len(b)
print b"""
