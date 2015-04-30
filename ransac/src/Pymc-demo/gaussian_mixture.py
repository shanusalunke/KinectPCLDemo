#import sys;
#sys.path.insert(0, "/Users/jscholz/vc/pymc/build/lib.macosx-10.9-intel-2.7/")

import numpy as np
import pymc
from matplotlib import pyplot
from matplotlib.mlab import normpdf

def plot_normal(mu, std, n=100):
    try:
        k, j = len(mu), len(std)
        assert(k==j)
    except:
        mu, std = [mu], [std]

    k = len(mu)
    xmin = min(mu) - 5*max(std)
    xmax = max(mu) + 5*max(std)
    x = np.linspace(xmin,xmax,n)

    Y = []
    for i in range(k):
        Y.append(normpdf(x,mu[i],std[i]))

    pyplot.interactive(True)
    for y in Y:
        pyplot.plot(x, y)

def generate_data(mu, sigma, pi, n):
    k = len(mu)
    z = np.random.choice(k, n, p=pi)
    return np.random.normal(mu[z], sigma[z])

def build_model(y, k, tau=None):
    '''
    :param y: observed data (1-D np.array)
    :param k: number of latent components (int)
    :param tau: component precisions.  If None, fit with a Gamma prior
    '''

    ## prior on component means
    lb = np.array([80])
    ub = np.array([200])
    lbc = np.tile(lb, k)
    ubc = np.tile(ub, k)
    #cheat
    # mix_mu = pymc.Lambda('mix_mu', lambda X=0: np.array([80, 130, 160]))
    mix_mu = pymc.Uniform('mix_mu', lbc, ubc)
    if tau == None:
        mix_tau = pymc.Gamma('mix_tau', alpha=1, beta=0.001, size=k) # this should work but sampler has trouble
    else:
        # mix_tau = pymc.Lambda('mix_tau', lambda X=0: tau)
        pymc.Uniform('mix_tau', lbc, ubc)

    ## categorical variable for mixture weights
    #pi = np.ones(k)/k   # uniform prior on component indicator
    pi = pymc.Dirichlet('pi', theta=np.ones(k)) # symmetric weak dirichlet on mixing props
    n = len(y)          # number of observations
    Z = pymc.Categorical('Z', size=n, p=pi) # latent component selector

    ## gaussian likelihood
    Y = pymc.Normal('Y', mu=mix_mu[Z], tau=mix_tau[Z], observed=True, value=y)
    return pymc.Model(locals())

if __name__ == '__main__':
    mu = np.array([100, 130, 160])               # component means
    sigma = np.array([0.8, 0.3, 1])       # component variances
    pi = np.array([0.5, 0.25, 0.25])        # non-uniform mixing weights
    #pi = np.ones(mu.shape[0])/mu.shape[0]  # uniform mixing weights
    n = 20                                  # number of points to generate
    # y = generate_data(mu, sigma, pi, n)     # generated data
    # y = np.array([ 8.26624296, -7.7384274 ,  0.33170278,  7.03139425, -8.09528804,
    #    -6.84145162,  7.51867937, -8.39869481,  7.95695608, -8.42421028,
    #    -7.89697929, -8.61046818, -8.28085273,  7.62077268, -0.55129317,
    #    -8.20013778,  0.6233622 , -7.15357652, -8.26254456,  0.02669849])

    # y = np.array([89.489348133314579, 105.03485388021663, 103.82024355732815, 91.504131030847802, 146.64458530427584, 129.37977842755956, 130.42276692736462, 170.58696241511308, 121.06452366220911, 140.44211731437332, 162.92599965310674, 120.31290994548047, 112.2679232778658, 152.22614664286178, 170.66495602344429, 118.43345482451321, 114.78335139061392])

    y = np.array([89.489348133314579, 105.03485388021663, 103.82024355732815, 91.504131030847802, 146.64458530427584, 129.37977842755956, 130.42276692736462, 170.58696241511308, 121.06452366220911, 140.44211731437332, 162.92599965310674, 120.31290994548047, 152.22614664286178,  118.43345482451321, 114.78335139061392])

    k = mu.shape[0] # number of components
    model = build_model(y, k) #, tau=1./sigma**2 make pymc model object
    mcmc = pymc.MCMC(model) # make an sampler for the model

    # run the sampler
    niter = 100000
    burn  = 90000
    thin  = 10
    tune_interval = 1000
    mcmc.sample(niter, burn, thin, tune_interval)

    # extract the sampler statistics
    fit = mcmc.stats()
    z_post_mean = fit['Z']['mean']              # posterior component indicators
    mu_post_mean = fit['mix_mu']['mean']        # posterior component means
    tau_post_mean = fit['mix_tau']['mean']      # posterior component precisions
    pi_post_mean = fit['pi']['mean']            # posterior component mixing coefficients
    print "\nPosterior Component Means:", mu_post_mean
    print "Posterior Component Precisions:", tau_post_mean
    print "Posterior Mixing Coefficients:", np.hstack([pi_post_mean, 1.-sum(pi_post_mean)])

    # pyplot.interactive(True)
    # pymc.Matplot.plot(mcmc)
    # mcmc.summary()
    pyplot.hist(y, bins=50, normed=1)
    # plot_normal(mu_post_mean, np.sqrt(1./tau_post_mean))
    plot_normal(mu_post_mean, np.sqrt(1./tau_post_mean))
    pyplot.show()

    import ipdb; ipdb.set_trace()
