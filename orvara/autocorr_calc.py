## This code is adapted from the emcee documentation. 
## It produces a plot showing a chain's estimated autocorrelation
## lengths against the length of the chain itself.


import os
import matplotlib.pyplot as plt
import numpy as np
import emcee
from astropy.io import fits
from orvara.format_fits import burnin_chain



# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1


def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf
    
def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

# Following the suggestion from Goodman & Weare (2010)
def autocorr_gw2010(y, c=5.0):
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


def autocorr_estimator(chain, output_dir, title):
    """
    Create a plot of estimated ACL versus chain length.
    Desired behavior: the blue and orange curves flatten
    for sufficiently long chain length, and the ACL value
    they flatten at approximates the true ACL. If they
    don't flatten, you need longer chains.

    chain: MCMC chain, pre-formatted in the orbit_plots module
    burning: # steps to burn in for
    output_dir: Save location for plot
    title: System name
    """

    #burnin = 100
    #chain_table = fits.open(chain_path)[1].data
    #chain_prestack = burnin_chain(chain_table.columns, burnin, reshape=False) # Don't reshape to Orvara default
    #chain_stacked = np.stack(chain_prestack) # Stack to get shape (nwalkers, nparams, nsteps)
    
    
    
    nparams_to_plot = int(np.shape(chain)[1]/3)
    for ind in range(nparams_to_plot):
        
        chain_single = chain[:,ind,:] # Take nth param only from stacked chains

        # Compute the estimators for a few different chain lengths
        N = np.exp(np.linspace(np.log(100), np.log(chain_single.shape[1]), 10)).astype(int) # From 100 up to chain length
        gw2010 = np.empty(len(N))
        new = np.empty(len(N))
        for i, n in enumerate(N):
            gw2010[i] = autocorr_gw2010(chain_single[:, :n])
            new[i] = autocorr_new(chain_single[:, :n])

        # import pdb; pdb.set_trace()

        # Plot the comparisons
        # plt.loglog(N, gw2010, "o-", label="G&W 2010") # Don't bother with GW bc docs say 'new' method is better
        plt.loglog(N, new, "o-", label=f"new_{ind}")
    ylim = plt.gca().get_ylim()
    plt.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
        
    plt.ylim(ylim)
    plt.xlabel("number of samples, $N$")
    plt.ylabel(r"$\tau$ estimates")

    #import pdb; pdb.set_trace()
    #plt.savefig('autocorr.png', dpi=200)
    plt.tight_layout()
    plt.title(title)
    plt.savefig(os.path.join(output_dir, 'autocorr_' + title)+'.png', dpi=250)
    #import pdb; pdb.set_trace()
    plt.close()
    
    return
    
if __name__=="__main__":
    autocorr_estimator("HD8765_Temp0_chain000.fits")
    
    
    
    

