#!/usr/bin/env python
"""
Orbit fitting code. The run function is the console entry point,
accessed by calling fit_orbit from the command line.
"""

from __future__ import print_function
import numpy as np
import os
import time
import emcee
from ptemcee import Sampler as PTSampler
from configparser import ConfigParser
from astropy.io import fits
from htof.main import Astrometry
from astropy.time import Time
from astropy.coordinates import Angle
import sys
import re
import random
from orvara import orbit
from orvara.config import parse_args
from orvara.format_fits import make_header, pack_cols
import pkg_resources

import scipy

_loglkwargs = {}

def set_initial_parameters(start_file, ntemps, nplanets, nwalkers, priors, njit=1,
                           minjit=-20, maxjit=20):
    
    par0 = np.ones((ntemps, nwalkers, 2 + 7 * nplanets))

    if start_file.lower() == 'none':
        mpri = 1
        jit = 0.5
        sau = 10
        esino = 0.5
        ecoso = 0.5
        inc = 1
        asc = 1
        lam = 1
        msec = 0.1

        sig = np.ones((ntemps, nwalkers, 2 + 7 * nplanets))*0.5    
        init = [jit, mpri]
        for i in range(nplanets):
             init += [msec, sau, esino, ecoso, inc, asc, lam]
        par0 *= np.asarray(init)

    else:
        init, sig = np.loadtxt(start_file).T
        init[0] = max(2*np.log10(init[0]), minjit) # Convert from m/s to units used in code
        try:
            par0 *= init
        except:
            raise ValueError('Starting file %s has the wrong format/length.' % (start_file))

    #######################################################################
    # Introduce scatter.  Normal in most parameters, lognormal in
    # mass and semimajor axis.
    #######################################################################
    
    # Generate random samples from N(0,1), reshape them to match the shape of par0, and rescale to the values in sig. These are the normal (not log-normal) errors.
    scatter = sig*np.random.randn(np.prod(par0.shape)).reshape(par0.shape)

    par0 += scatter # Offset the median values by the scatter

    # Judah: omit lines below to sample Msec and a from a normal dist. instead of a log-normal dist.
    # Now subtract off the linear scatter for the 'Msec' and 'a' parameters, and instead multiply the values by the exponential of that linear scatter.
    #par0[..., 2::7] = (par0[..., 2::7] - scatter[..., 2::7])*np.exp(scatter[..., 2::7])
    #par0[..., 3::7] = (par0[..., 3::7] - scatter[..., 3::7])*np.exp(scatter[..., 3::7])
    #import pdb; pdb.set_trace()
    # Ensure that values are within allowable ranges.
    
    """
    bounds = [[0, minjit, maxjit],   # jitter
              [1, 1e-4, 1e3],        # mpri (Solar masses)
              [2, 1e-4, 1e3],        # msec (Solar masses)
              [3, 1e-5, 2e5],        # semimajor axis (AU)
              [6, 1e-5, np.pi],      # inclination (radians)
              [7, -np.pi, 3*np.pi],  # longitude of ascending node (rad)
              [8, -np.pi, 3*np.pi]]  # long at ref epoch (rad)
    """

    # Start with bounds for jitter and mpri
    bounds = [[0, minjit, maxjit],   # jitter
              [1, 1e-4, 1e3]]        # mpri (Solar masses)

    # Judah: add bounds for each planet. This makes sure all initial guesses are within prior constraints.
    for i in range(nplanets):
        min_msec = priors['min_msec{}'.format(i)]
        min_a = priors['min_a{}'.format(i)]
        max_a = priors['max_a{}'.format(i)]

        min_msec = max(1e-5, min_msec) # Make sure msec is at least 1e-5
        pl_bounds = [[7*i+2, min_msec, 1], # msec
                     [7*i+3, min_a, max_a], # SMA
                     [7*i+6, 1e-5, np.pi],  # inc.
                     [7*i+7, -np.pi, 3*np.pi], # \Omega
                     [7*i+8, -np.pi, 3*np.pi]] # Mean long. at 2010.0

        #if i==2:
            #import pdb; pdb.set_trace()
        bounds += pl_bounds # Tack on the companion-specific bounds
        
    # Judah: apply bounds for each planet to make sure all initial guesses are within prior constraints.
    for i in range(len(bounds)):

        j, minval, maxval = bounds[i]
        if j <= 1:
            par0[..., j][par0[..., j] < minval] = minval
            par0[..., j][par0[..., j] > maxval] = maxval
        else: # Made this exactly the same as above. This makes each planet have its own SMA bounds
            par0[..., j][par0[..., j] < minval] = minval
            par0[..., j][par0[..., j] > maxval] = maxval


    #import pdb; pdb.set_trace()
            
    # Eccentricity is a special case.  Cap at 0.99.
    ecc = par0[..., 4::7]**2 + par0[..., 5::7]**2 # Calc ecc from sesinw and secosw
    fac = np.ones(ecc.shape)
    fac[ecc > 0.99] = np.sqrt(0.99)/np.sqrt(ecc[ecc > 0.99]) # For ecc>0.99, fac is sqrt(0.99)/sqrt(>0.99)
    par0[..., 4::7] *= fac # Start at element 4 and mult every 7th element by fac
    par0[..., 5::7] *= fac # Start at element 5 and mult every 7th element by fac

    # Move jitter to the end, add (shuffled) realizations if needed.
    par0_jitlast = np.zeros((ntemps, nwalkers, par0.shape[-1] + njit - 1))
    par0_jitlast[..., :-njit] = par0[..., 1:]
    for i in range(njit):
        random.shuffle(par0[..., 0].T)
        par0_jitlast[..., -1 - i] = par0[..., 0]


    #import pdb; pdb.set_trace()

    return par0_jitlast


def initialize_data(config, companion_gaia):
    # load in items from the ConfigParser object
    HipID = config.getint('data_paths', 'HipID', fallback=0)
    HGCAFile = config.get('data_paths', 'HGCAFile')
    if not os.path.exists(HGCAFile):
        raise FileNotFoundError(f'No HGCA file found at {HGCAFile}')

    try:
        table = fits.open(HGCAFile)[1].data
        epoch_ra_gaia = table[table['hip_id'] == 1]['epoch_ra_gaia']
        if np.isclose(epoch_ra_gaia, 2015.60211565):
            HGCAVersion = 'GaiaDR2'
            gaia_mission_length_yrs = 1.75
        elif np.isclose(epoch_ra_gaia, 2015.92749023):
            HGCAVersion = 'GaiaeDR3'
            gaia_mission_length_yrs = 2.76
            if 'gaia_npar' in table.names:
                # TODO change this to GaiaDR3 when HTOF has a 'GaiaDR3' parser.
                #  right now for testing, gaiaedr3 works fine because the baseline of dr3 and edr3 will be the same.
                HGCAVersion = 'GaiaeDR3'
                gaia_mission_length_yrs = 2.76
        else:
            raise ValueError("Cannot match %s to either DR2 or eDR3, or DR3 based on RA epoch of Gaia" % (HGCAFile))
    except:
        raise ValueError("Cannot access HIP 1 in HGCA file" + HGCAFile)

    RVFile = config.get('data_paths', 'RVFile', fallback='')
    relRVFile = config.get('data_paths', 'relRVFile', fallback='')
    AstrometryFile = config.get('data_paths', 'AstrometryFile', fallback='')
    GaiaDataDir = config.get('data_paths', 'GaiaDataDir', fallback='')
    Hip2DataDir = config.get('data_paths', 'Hip2DataDir', fallback='')
    Hip1DataDir = config.get('data_paths', 'Hip1DataDir', fallback='')
    use_epoch_astrometry = config.getboolean('mcmc_settings', 'use_epoch_astrometry', fallback=False)
    data = orbit.Data(HipID, HGCAFile, RVFile, AstrometryFile, relRVFile=relRVFile, companion_gaia=companion_gaia,
                      gaia_mission_length_yrs=gaia_mission_length_yrs)
    if use_epoch_astrometry and data.use_abs_ast == 1:
        # five-parameter fit means a first order polynomial, 7-parameter means 2nd order polynomial etc..
        gaia_fit_degree = {5: 1, 7: 2, 9: 3}[data.gaia_npar]
        Gaia_fitter = Astrometry(HGCAVersion, '%06d' % (HipID), GaiaDataDir,
                                 central_epoch_ra=data.epRA_G,
                                 central_epoch_dec=data.epDec_G,
                                 format='jyear', fit_degree=gaia_fit_degree,
                                 use_parallax=True, use_catalog_parallax_factors=True)
        Hip2_fitter = Astrometry('Hip2or21', '%06d' % (HipID), Hip2DataDir,
                                 central_epoch_ra=data.epRA_H,
                                 central_epoch_dec=data.epDec_H,
                                 format='jyear', use_parallax=True, use_catalog_parallax_factors=True)
        Hip1_fitter = Astrometry('Hip1', '%06d' % (HipID), Hip1DataDir,
                                 central_epoch_ra=data.epRA_H,
                                 central_epoch_dec=data.epDec_H,
                                 format='jyear', use_parallax=True, use_catalog_parallax_factors=True)
        # instantiate C versions of the astrometric fitter which are much faster than HTOF's Astrometry
        hip1_fast_fitter = orbit.AstrometricFitter(Hip1_fitter)
        hip2_fast_fitter = orbit.AstrometricFitter(Hip2_fitter)
        gaia_fast_fitter = orbit.AstrometricFitter(Gaia_fitter)
        # generate the data object.
        data = orbit.Data(HipID, HGCAFile, RVFile, AstrometryFile, 
                          use_epoch_astrometry=use_epoch_astrometry,
                          relRVFile=relRVFile,
                          epochs_Hip1=Hip1_fitter.data.julian_day_epoch(),
                          epochs_Hip2=Hip2_fitter.data.julian_day_epoch(),
                          epochs_Gaia=Gaia_fitter.data.julian_day_epoch(),
                          companion_gaia=companion_gaia, verbose=False)
    elif data.use_abs_ast == 1:
        hip1_fast_fitter, hip2_fast_fitter, gaia_fast_fitter = None, None, None
    else:
        hip1_fast_fitter, hip2_fast_fitter, gaia_fast_fitter = None, None, None
        try:
            data.plx = 1e-3*config.getfloat('priors_settings', 'parallax')
            data.plx_err = 1e-3*config.getfloat('priors_settings', 'parallax_error')
        except:
            raise RuntimeError("Cannot load absolute astrometry. Please supply a prior "
                               "for parallax and parallax_error in the priors_settings area"
                               " of the configuration file.")
    # If at any point, we implement jerk fitting to the calc_pms_no_epoch_astrometry(), this catch can be removed:
    if data.gaia_npar == 9 and not use_epoch_astrometry:
        raise RuntimeError("This is a 9-parameter source in Gaia, but you have set use_epoch_astrometry=False"
                           " in the configuration file. Please enable use_epoch_astrometry=True for this source, and "
                           "follow the directions in section 'Epoch Astrometry' of the readme.")
    return data, hip1_fast_fitter, hip2_fast_fitter, gaia_fast_fitter


def lnprob(theta, returninfo=False, RVoffsets=False, use_epoch_astrometry=False,
           data=None, nplanets=1, H1f=None, H2f=None, Gf=None, priors=None, 
           njitters=1):
    """
    Log likelihood function for the joint parameters
    :param theta:
    :param returninfo:
    :param use_epoch_astrometry:
    :param data:
    :param nplanets:
    :param H1f:
    :param H2f:
    :param Gf:
    :return:
    """
    #import pdb; pdb.set_trace()
    model = orbit.Model(data)
    lnp = 0
    for i in range(nplanets):
        # Note that params.mpri is really the mass contained in the primary + companions inner to the current planet.
        # params.mpri_true is the real mass of the primary. So params.mpri should really be renamed params.interior_mass
        params = orbit.Params(theta, i, nplanets, data.nInst, njitters)
        lnp = lnp + orbit.lnprior(params, minjit=priors['minjit'],
                                          maxjit=priors['maxjit'],
                                          min_msec=priors['min_msec{}'.format(i)],
                                          max_msec=priors['max_msec{}'.format(i)], 
                                          min_a=priors['min_a{}'.format(i)],
                                          max_a=priors['max_a{}'.format(i)],
                                          min_ecc=priors['min_ecc{}'.format(i)],
                                          max_ecc=priors['max_ecc{}'.format(i)],)

        #import pdb; pdb.set_trace()
        #if 0.104794<params.msec<0.104795:
            #import pdb; pdb.set_trace()
        #if np.isfinite(lnp):
            #import pdb; pdb.set_trace()
        if not np.isfinite(lnp):
            model.free()
            params.free()
            #import pdb; pdb.set_trace()
            return -np.inf

        orbit.calc_EA_RPP(data, params, model)
        orbit.calc_RV(data, params, model)
        if data.n_rel_RV > 0:
            # slightly more performant if we ignore the relative RV calculation call entirely if
            # we do not have any relative RVs to calculate.
            orbit.calc_relRV(data, params, model, i)
        orbit.calc_offsets(data, params, model, i)

        chisq_sec = (params.msec - priors[f'm_secondary{i}'])**2/priors[f'm_secondary{i}_sig']**2
        if chisq_sec > 0:  # If chisq_sec > 0 then a prior was set on the mass of the secondary
            # undo the default 1/m prior that is set in orbit.pyx (see lnprior())
            lnp = lnp - 0.5*chisq_sec + np.log(params.msec)
        # else, don't change lnp at all.

        # Free params if we need to cycle through the next companion
        if i < nplanets - 1:
            params.free()

    if use_epoch_astrometry and data.use_abs_ast:
        orbit.calc_PMs_epoch_astrometry(data, model, H1f, H2f, Gf)
    elif data.use_abs_ast:
        orbit.calc_PMs_no_epoch_astrometry(data, model)

    if returninfo:
        return orbit.calcL(data, params, model, chisq_resids=True, RVoffsets=RVoffsets)

    if np.isfinite(priors['mpri_sig']):
        return lnp - 0.5*(params.mpri_true - priors['mpri'])**2/priors['mpri_sig']**2 + orbit.calcL(data, params, model)
    else:
        return lnp - np.log(params.mpri_true) + orbit.calcL(data, params, model)

    
def return_one(theta):
    return 1.


def avoid_pickle_lnprob(theta):
    global _loglkwargs
    return lnprob(theta, **_loglkwargs)


def get_priors(config):
    priors = {}
    priors['mpri'] = config.getfloat('priors_settings', 'mpri', fallback=1.)
    priors['mpri_sig'] = config.getfloat('priors_settings', 'mpri_sig', fallback=np.inf)
    # priors on the masses of the companions (labeled 0 though 9). Limit to 10 planet systems.
    # Not sure why we go up to 10. We know num_companions from the config file. Keeping for consistency.
    for i in range(10):
        priors[f'm_secondary{i}'] = config.getfloat('priors_settings', f'm_secondary{i}', fallback=1.)
        priors[f'm_secondary{i}_sig'] = config.getfloat('priors_settings', f'm_secondary{i}_sig', fallback=np.inf)
        priors['min_msec{}'.format(i)] = config.getfloat('priors_settings', 'min_msec{}'.format(i), fallback=0)
        priors['max_msec{}'.format(i)] = config.getfloat('priors_settings', 'max_msec{}'.format(i), fallback=1)
        priors['min_a{}'.format(i)] = config.getfloat('priors_settings', 'min_a{}'.format(i), fallback=1)
        priors['max_a{}'.format(i)] = config.getfloat('priors_settings', 'max_a{}'.format(i), fallback=100)
        priors['min_ecc{}'.format(i)] = config.getfloat('priors_settings', 'min_ecc{}'.format(i), fallback=0.)
        priors['max_ecc{}'.format(i)] = config.getfloat('priors_settings', 'max_ecc{}'.format(i), fallback=0.99)
    # priors on the RV jitter. Converting from m/s to orvara internal units.
    priors['minjit'] = config.getfloat('priors_settings', 'minjitter', fallback = 1e-5)
    priors['minjit'] = max(priors['minjit'], 1e-20) # effectively zero, but we need the log
    priors['minjit'] = 2*np.log10(priors['minjit'])
    priors['maxjit'] = config.getfloat('priors_settings', 'maxjitter', fallback = 1e3)
    priors['maxjit'] = 2*np.log10(priors['maxjit'])

    if priors['maxjit'] < priors['minjit']:
        raise ValueError("Requested maximum jitter < minimum jitter")
    return priors


def get_gaia_catalog_companion(config):
    companion_gaia = {}
    companion_gaia['ID'] = config.getint('secondary_gaia', 'companion_ID', fallback=-1)
    companion_gaia['pmra'] = config.getfloat('secondary_gaia', 'pmra', fallback=0)
    companion_gaia['pmdec'] = config.getfloat('secondary_gaia', 'pmdec', fallback=0)
    companion_gaia['e_pmra'] = config.getfloat('secondary_gaia', 'epmra', fallback=1)
    companion_gaia['e_pmdec'] = config.getfloat('secondary_gaia', 'epmdec', fallback=1)
    companion_gaia['corr_pmra_pmdec'] = config.getfloat('secondary_gaia', 'corr_pmra_pmdec', fallback=0)
    return companion_gaia


def run():
    """
    Initialize and run the MCMC sampler.
    """
    global _loglkwargs
    start_time = time.time()

    args = parse_args()
    config = ConfigParser()
    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f'No config file found at {args.config_file}')
    if not os.path.exists(args.output_dir):
        raise FileNotFoundError(f'No output_dir found at {args.output_dir}')
    config.read(args.config_file)

    # set the mcmc parameters
    nwalkers = config.getint('mcmc_settings', 'nwalkers', fallback=100)
    ntemps = config.getint('mcmc_settings', 'ntemps', fallback=10)
    nplanets = config.getint('mcmc_settings', 'nplanets')
    jit_per_inst = config.getboolean('mcmc_settings', 'jit_per_inst', fallback=False)
    nstep = config.getint('mcmc_settings', 'nstep')
    thin = config.getint('mcmc_settings', 'thin', fallback=50)
    nthreads = config.getint('mcmc_settings', 'nthreads', fallback=1)
    use_epoch_astrometry = config.getboolean('mcmc_settings', 'use_epoch_astrometry', fallback=False)
    HipID = config.getint('data_paths', 'HipID', fallback=0)
    start_file = config.get('data_paths', 'start_file', fallback='none')
    starname = config.get('plotting', 'target', fallback='HIP%d' % (HipID))
    #
    priors = get_priors(config)
    companion_gaia = get_gaia_catalog_companion(config)
    
    # Save configuration file in FITS header format
    header = make_header(args.config_file)

    data, H1f, H2f, Gf = initialize_data(config, companion_gaia)

    # set initial conditions
    if jit_per_inst:
        njit = data.nInst
    else:
        njit = 1
    #import pdb; pdb.set_trace()
    par0 = set_initial_parameters(start_file, ntemps, nplanets, nwalkers, priors, 
                                  njit=njit, minjit=priors['minjit'], 
                                  maxjit=priors['maxjit'])
    ndim = par0[0, 0, :].size

    # set arguments for emcee PTSampler and the log-likelyhood (lnprob)
    samplekwargs = {'thin': thin}
    loglkwargs = {'returninfo': False, 'use_epoch_astrometry': use_epoch_astrometry,
                  'data': data, 'nplanets': nplanets, 'H1f': H1f, 'H2f': H2f, 'Gf': Gf, 'priors': priors, 'njitters': njit}
    _loglkwargs = loglkwargs
    # run sampler without feeding it loglkwargs directly, since loglkwargs contains non-picklable C objects.

    use_dynesty=True
    if use_dynesty:

        #import pdb; pdb.set_trace()

        #def ptform(u): # Most simple: uniform prior
            #return u

        import dynesty
        #import pdb; pdb.set_trace()
        #sample0 = dynesty.NestedSampler(avoid_pickle_lnprob, dynesty_prior, ndim)
        dsampler = dynesty.DynamicNestedSampler(avoid_pickle_lnprob, dynesty_prior, ndim, bound='multi')

        print('Using dynesty')
        dsampler.run_nested()
        dresults = dsampler.results
        sdfsd
        

    else:
        try:
            use_ptemcee = False
            sample0 = emcee.PTSampler(ntemps, nwalkers, ndim, avoid_pickle_lnprob, return_one, threads=nthreads)
            print('Using emcee.PTSampler.')
        except:
            use_ptemcee = True
            sample0 = PTSampler(ntemps=ntemps, nwalkers=nwalkers, dim=ndim,
                            logl=avoid_pickle_lnprob, logp=return_one,
                            threads=nthreads)
        print('Using ptemcee.')

        print("Running MCMC ... ")
        #sample0.run_mcmc(par0, nstep, **samplekwargs)
        #add a progress bar
        width = 30
        N = min(100, nstep//thin)
        n_taken = 0
        sys.stdout.write("[{0}]  {1}%".format(' ' * width, 0))
        for ipct in range(N):
            dn = (((nstep*(ipct + 1))//N - n_taken)//thin)*thin
            n_taken += dn
            #import pdb; pdb.set_trace()
            if ipct == 0:
                sample0.run_mcmc(par0, dn, **samplekwargs)
            else:
                # Continue from last step
                sample0.run_mcmc(sample0.chain[..., -1, :], dn, **samplekwargs)
            n = int((width+1) * float(ipct + 1) / N)
            sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
            sys.stdout.write("%3d%%" % (int(100*(ipct + 1)/N)))
        sys.stdout.write("\n")
        
        print('Total Time: %.0f seconds' % (time.time() - start_time))
        print("Mean acceptance fraction (cold chain):     {0:.6f}".format(np.mean(sample0.acceptance_fraction[0, :])))
        #import pdb; pdb.set_trace()




    ## Judah change: iterate over multiple chains corresp. to different Temps. Save a handful
    save_highT = True
    if save_highT:
        num_temps = sample0.logprobability.shape[0] # Assume use_ptemcee==True
        temp_ind_list = [0, num_temps-1] # Just lowT and highT chains
    else:
        temp_ind_list = [0] # Only the lowest temp (default behavior)

    for temp_ind in temp_ind_list:

        sample0_chain = sample0.chain[temp_ind] # Select the chain of the desired temperature
        
        # save data
        if not use_ptemcee:
            sample0_lnprob = sample0.lnprobability[temp_ind]
            shape = sample0_lnprob.shape
        else:
            sample0_logprob = sample0.logprobability[temp_ind]
            shape = sample0_logprob.shape


        parfit = np.zeros((shape[0], shape[1], 9 + data.nInst))
    
        loglkwargs['returninfo'] = True
        loglkwargs['RVoffsets'] = True
    
        for i in range(shape[0]):
            for j in range(shape[1]):
                res, RVoffsets = lnprob(sample0_chain[i, j], **loglkwargs)
                parfit[i, j, :9] = [res.plx_best, res.pmra_best, res.pmdec_best,
                                    res.chisq_sep, res.chisq_PA,
                                    res.chisq_H, res.chisq_HG, res.chisq_G, res.chisq_relRV]
            
                if data.nInst > 0:
                    parfit[i, j, 9:] = RVoffsets

        colnames = ['mpri']
        units = ['msun']
        for i in range(nplanets):
            colnames += [s + '%d' % (i) for s in ['msec', 'sau', 'esino', 'ecoso',
                                              'inc', 'asc', 'lam']]
            units += ['msun', 'au', '', '', 'radians', 'radians', 'radians']

        if njit == 1:
            colnames += ['jitter']
            units += ['m/s']
            sample0.chain[0][..., -1] = 10**(0.5*sample0_chain[..., -1])
        else:
            for i in range(njit):
                colnames += ['jitter%d' % (i)]
                units += ['m/s']
            sample0_chain[..., -njit:] = 10**(0.5*sample0_chain[..., -njit:])

        colnames += ['lnp']
        colnames += ['plx_ML', 'pmra_ML', 'pmdec_ML', 'chisq_sep', 
                     'chisq_PA', 'chisq_H', 'chisq_HG', 'chisq_G', 'chisq_relRV']
        units += ['', 'arcsec', 'arcsec/yr', 'arcsec/yr', '', '', '', '', '', '']
        colnames += ['RV_ZP_%d_ML' % (i) for i in range(data.nInst)]
        units += ['m/s' for i in range(data.nInst)]

        out = fits.HDUList(fits.PrimaryHDU(None, header))

        if not use_ptemcee:
            lnp = sample0_lnprob
        else:
            lnp = sample0_logprob

        out.append(pack_cols(sample0_chain, lnp, parfit, colnames, units))

        for i in range(1000):
            #filename = os.path.join(args.output_dir, '%s_chain%03d.fits' % (starname, i))
            #import pdb; pdb.set_trace()
            filename = os.path.join(args.output_dir, '{}_Temp{}_chain{}.fits'.format(starname, temp_ind, '{}'.format(i).zfill(3)))
            if not os.path.isfile(filename):
                print('Writing output to {0}'.format(filename))
                out.writeto(filename, overwrite=False)
                break

    return out

def dynesty_prior(u):
    """
    Transforms draws from an n-dimensional
    unit cube, Unif[0,1), into draws from
    the appropriate prior distributions.

    There should be 2 + 7*n parameters, where
    n is the number of planets in the model.
    The first 2 params are jitter and Mprimary
    """
    x = np.array(u) # Copy of u

    # First, get the priors from the config file
    args = parse_args() # Parses arguments to the original script call
    config = ConfigParser()
    config.read(args.config_file)
    priors = get_priors(config)
    nplanets = config.getint('mcmc_settings', 'nplanets')


    ## Jitter: use hard log-limits
    ## Note: jit was moved to the END in set_initial_parameters
    x[-1] = u[-1]*(priors['maxjit']-priors['minjit']) + priors['minjit']

    ## Msec: normal
    x[0] = scipy.stats.norm.ppf(u[1], loc=priors['mpri'], scale=priors['mpri_sig'])


    for i in range(nplanets):
        #import pdb; pdb.set_trace()
        ## Msec: hard limits
        x[7*i+1] = u[7*i+1]*(priors[f'max_msec{i}']-priors[f'min_msec{i}']) + priors[f'min_msec{i}']

        ## SMA
        x[7*i+2] = u[7*i+2]*(priors[f'max_a{i}']-priors[f'min_a{i}']) + priors[f'min_a{i}']

        ## sesino and secoso are simply uniform between [0,1)
        x[7*i+3] = u[7*i+3]
        x[7*i+4] = u[7*i+4]

        ## cos(i) should be uniform, so i should be cos^-1(u), where u is drawn uniformly
        x[7*i+5] = np.arccos(u[7*i+5])

        ## \Omega, and \lambda are simply uniform between [0,2pi)
        x[7*i+6] = u[7*i+6]*2*np.pi
        x[7*i+7] = u[7*i+7]*2*np.pi


    return x









if __name__ == "__main__":
    run()

















