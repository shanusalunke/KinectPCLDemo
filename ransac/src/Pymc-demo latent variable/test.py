import numpy as np
import scipy
import pymc
from matplotlib import pyplot
from matplotlib.mlab import normpdf


def linear_model(a, b, x, sigma):
    return np.random.normal(a*x + b, sigma)


def quadratic_model(a, b, c, x, sigma):
    return np.random.normal(a*x**2 + b*x + c, sigma)


# def generate_linear_data(a, b, v, sigma):
#     return linear_model(a, b, v, sigma)


def build_linear_model(volume_data, mass_data):
    '''
    :param v: volume (observed data)
    :param m: mass (observed data)
    '''

    # weak gaussian prior on linear regression coefficient
    a = pymc.Normal('a', mu=0, tau=0.1)

    # weak gaussian prior on linear regression intercept
    b = pymc.Normal('b', mu=0, tau=0.1)

    @pymc.deterministic
    def muM(a=a, b=b, v=volume_data):
        '''
        Implements a linear regression model for mass on volume.
        '''
        return a * v + b

    # weak uniform prior on output noise variance
    sigma = pymc.Uniform('sigma', 0.0, 200.0, value=3.0) # value=10., observed=True

    # overall model likelihood for m ~ a*v + b + noise, noise ~ Normal(0, sigma^2)
    m = pymc.Normal('m', mu=muM, tau=1.0/sigma**2, value=mass_data, observed=True)

    return pymc.Model(locals())

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

def generate_data(mu_m, mu_v, sigma_m, sigma_v, pi, n):
    k = len(mu_m)
    z = np.random.choice(k, n, p=pi)
    return np.random.normal(mu_m[z], sigma_m[z]), np.random.normal(mu_v[z], sigma_v[z])

def build_latent_model(m, v, k, tau=None):
    '''
    :param m: observed data (1-D np.array)
    :param k: number of latent components (int)
    :param tau: component precisions.  If None, fit with a Gamma prior
    '''

    ## prior on component means
    lb = np.array([0])
    ub = np.array([40])
    lbc = np.tile(lb, k)
    ubc = np.tile(ub, k)

    # legit version:
    mu_m = pymc.Uniform('mu_m', lower=lbc, upper=ubc)
    mu_v = pymc.Uniform('mu_v', lower=lbc, upper=ubc)

    # cheating version:
    # mu_m = pymc.Lambda('mu_m', lambda X=0: np.array([3.0, 5.0, 7.0]))
    # mu_v = pymc.Lambda('mu_v', lambda X=0: np.array([12.0, 16.0, 30.0]))

    if tau == None:
        tau_m = pymc.Gamma('tau_m', alpha=1, beta=0.001, size=k) # this should work but sampler has trouble
        tau_v = pymc.Gamma('tau_v', alpha=1, beta=0.001, size=k) # this should work but sampler has trouble
    else:
        tau_m = pymc.Lambda('tau_m', lambda X=0: tau)
        tau_v = pymc.Lambda('tau_v', lambda X=0: tau)

    ## categorical variable for mixture weights
    #pi = np.ones(k)/k   # uniform prior on component indicator
    pi = pymc.Dirichlet('pi', theta=np.ones(k)) # symmetric weak dirichlet on mixing props
    n = len(m)          # number of observations
    Z = pymc.Categorical('Z', size=n, p=pi) # latent component selector

    ## gaussian likelihood
    M = pymc.Normal('M', mu=mu_m[Z], tau=tau_m[Z], observed=True, value=m)
    V = pymc.Normal('V', mu=mu_v[Z], tau=tau_m[Z], observed=True, value=v)

    return pymc.Model(locals())


if __name__ == '__main__':
    # v = np.linspace(1, 3, num=20)
    # m = generate_data(a=2.5, b=2, v=v, sigma=0.25)

    mu_m = np.array([3.0, 5.0, 7.0])          # component means
    mu_v = np.array([12.0, 16.0, 30.0])       # component means

    sigma_m = np.array([0.05, 0.05, 0.05])       # component variances
    sigma_v = np.array([0.05, 0.05, 0.05])       # component variances

    pi = np.array([0.5, 0.25, 0.25])        # non-uniform mixing weights
    #pi = np.ones(mu_m.shape[0])/mu_m.shape[0]  # uniform mixing weights
    n = 30                                  # number of points to generate
    m,v = generate_data(mu_m, mu_v, sigma_m, sigma_v, pi, n)     # generated data



    #Linear Model
    model_linear = build_linear_model(v, m)   # make pymc model object
    mcmc_linear = pymc.MCMC(model_linear)     # make an sampler for the model

    #Latent Model
    k = mu_m.shape[0] # number of components
    # model = build_model(m, v, k, tau=1./sigma_m**2) # make pymc model object
    model_latent = build_latent_model(m, v, k) # make pymc model object
    mcmc_latent = pymc.MCMC(model_latent) # make an sampler for the model


    # run the sampler
    niter = 10000
    burn  = 9000
    thin  = 10
    tune_interval = 1000
    mcmc_linear.sample(niter, burn, thin, tune_interval)
    mcmc_latent.sample(niter, burn, thin, tune_interval)


    observed_volume = np.array([15.5])

    #Linear
    # extract the sampler statistics
    fit = mcmc_linear.stats()
    a_post_mean = fit['a']['mean']          # posterior component means
    b_post_mean = fit['b']['mean']          # posterior component precisions
    sigma_post_mean = fit['sigma']['mean']  # posterior component mixing coefficients
    # print "\nPosterior linear coefficient:", a_post_mean
    # print "Posterior intercept:", b_post_mean
    # print "Posterior variance:", sigma_post_mean

    predicted_means = linear_model(
        a=a_post_mean, b=b_post_mean,
        x=observed_volume, sigma=sigma_post_mean)

    print "\nLinear: Given volume observations: ", observed_volume, " predicted: ", predicted_means


    #Latent
    # extract the sampler statistics
    fit = mcmc_latent.stats()
    z_post_mean = fit['Z']['mean']              # posterior component indicators
    mu_m_post_mean = fit['mu_m']['mean']        # posterior component means
    mu_v_post_mean = fit['mu_v']['mean']        # posterior component means
    tau_m_post_mean = fit['tau_m']['mean']      # posterior component precisions
    tau_v_post_mean = fit['tau_v']['mean']      # posterior component precisions
    pi_post_mean = fit['pi']['mean']            # posterior component mixing coefficients
    # print "\nPosterior Component Means for mass:", mu_m_post_mean
    # print "Posterior Component Means for volume:", mu_v_post_mean
    # print "Posterior Component Precisions for mass:", tau_m_post_mean
    # print "Posterior Component Precisions for volume:", tau_v_post_mean
    # print "Posterior Mixing Coefficients:", np.hstack([pi_post_mean, 1.-sum(pi_post_mean)])
    pi_post_mean_all = np.hstack([pi_post_mean, 1-sum(pi_post_mean)])
    map_component = np.argmax([np.log(pi_post_mean_all[i])+scipy.stats.norm.logpdf(15.5, loc=mu_v_post_mean[i], scale=1./tau_v_post_mean[i]) for i in range(len(mu_v_post_mean))])
    predicted_mass = mu_m_post_mean[map_component]

    print "Latent: Given volume observations: ", observed_volume, " predicted: ", predicted_mass


    # import ipdb; ipdb.set_trace()
