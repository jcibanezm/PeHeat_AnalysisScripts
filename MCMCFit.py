import emcee
import matplotlib.pyplot as plt
import numpy as np

def plot_data(x, y):
    plt.plot(x, y, "-k", linewidth=2, linestyle="steps", )
    #plt.xlim(-2.2,1.2)
    #plt.ylim(-52,-41)
    
    plt.xlabel("Z", fontsize=15)
    plt.ylabel("$\Gamma^{''}_{pe}(a, Z)$ [erg s$^{-1}$]", fontsize=13)

    plt.tick_params(axis='both', which='major', length=10, width=2,  labelsize=10, direction="in")
    plt.tick_params(axis='both', which='minor', length=5, width=1.5, labelsize=10, direction="in")

    return

def ln_likelihood(x, y, modelfunc, params):
    model_y = modelfunc(params, x)
    chi = (y - model_y)
    return -0.5*np.sum(chi**2)

BOUNDARIES = [(-100,100),(-100,100),(-100,100),(-100,100)]

def ln_prior(params, bound=BOUNDARIES):
    result = 0.0
    for i in range(len(params)):
        if params[i]<bound[i][0] or params[i]>bound[i][1] : result = -np.inf 
    return result


def lin_model(params, x):
    m, b = params
    return m*x + b

def lnprob(params, x, y):
    return ln_likelihood(x, y, lin_model, params) + ln_prior(params)


def quad_model(params, x):
    a, m, b = params
    return a*x**2 + m*x + b

def lnprob_quad(params, x, y):
    return ln_likelihood(x, y, quad_model, params) + ln_prior(params)

def LinearMCMC(x, y):
    """
    Use emcee to do a MCMC Bayesian parameter estimation of the heating as a function of charge.
    Linear fit to the 
    """
    # Linear regression
    NWALKERS = 100
    NDIMS    = 2
    theta0   = np.zeros(shape=(NWALKERS,NDIMS))

    ## Initialize around those values
    theta0[:,0] = np.random.uniform(-100.0,100.0,NWALKERS)
    theta0[:,1] = np.random.uniform(-100.0,100.0,NWALKERS)

    sampler = emcee.EnsembleSampler(NWALKERS, NDIMS, lnprob, args=[x,y])

    NBURN = 100
    theta, prob, state = sampler.run_mcmc(theta0, NBURN)

    sampler.reset()

    NSTEPS = 1000
    theta2, prob2, state2 = sampler.run_mcmc(theta, NSTEPS)

    median = np.percentile(sampler.flatchain, 50, axis=0)
    
    return median, sampler, theta


def QuadMCMC(x, y):
    
    NWALKERS = 100
    NDIMS    = 3

    quadsampler = emcee.EnsembleSampler(NWALKERS, NDIMS, lnprob_quad, args=(x,y))

    ## Initialize new walkers
    theta0 = np.zeros(shape=(NWALKERS,NDIMS))
    theta0[:,0] = np.random.uniform(-100,100,NWALKERS)
    theta0[:,1] = np.random.uniform(-100,100,NWALKERS)
    theta0[:,2] = np.random.uniform(-100,100,NWALKERS)

    NBURN = 1000
    theta, prob, state = quadsampler.run_mcmc(theta0, NBURN)
    
    quadsampler.reset()

    NSTEPS = 1000
    theta2, prob2, state2 = quadsampler.run_mcmc(theta, NSTEPS)

    median = np.quantile(quadsampler.flatchain, 0.5, axis=0)
    
    return median, quadsampler, theta2

