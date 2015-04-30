#import sys;
#sys.path.insert(0, "/Users/jscholz/vc/pymc/build/lib.macosx-10.9-intel-2.7/")

import numpy as np
import pymc
from matplotlib import pyplot
from matplotlib.mlab import normpdf

'''
Implements an alternative object model in which we
perform a density estimation on the observed data
rather than looking for functional relationships
(e.g. a linear effects model).

In this example we again consider just mass and volume,
but


------------------------------------------------------------------
all visual features:
v - bounding polygon volume
a - bounding polygon area
nu - normals for each detected plane
kappa - 3x3 covariance matrix of point cloud (mass distribution, like inertia matrix)

all physical feature:
m - mass
mu_x - friction in x direction
mu_x - friction in y direction
w_x - friction joint x
w_y - friction joint y
w_t - friction joint angle

'''

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

def build_model(m, v, k, tau=None):
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
    # mu_m = np.array([3.0, 5.0, 7.0])          # component means
    # mu_v = np.array([12.0, 16.0, 30.0])       # component means
    #
    # sigma_m = np.array([0.05, 0.05, 0.05])       # component variances
    # sigma_v = np.array([0.05, 0.05, 0.05])       # component variances
    #
    # pi = np.array([0.5, 0.25, 0.25])        # non-uniform mixing weights
    # #pi = np.ones(mu_m.shape[0])/mu_m.shape[0]  # uniform mixing weights
    # n = 30                                  # number of points to generate
    # m,v = generate_data(mu_m, mu_v, sigma_m, sigma_v, pi, n)     # generated data


    v = np.array([89.489348133314579, 105.03485388021663, 103.82024355732815, 91.504131030847802, 146.64458530427584, 129.37977842755956, 130.42276692736462, 170.58696241511308, 121.06452366220911, 140.44211731437332, 162.92599965310674, 120.31290994548047, 152.22614664286178,  118.43345482451321, 114.78335139061392])

    mu_m = np.array([1, 2, 3])               # component means
    mu_v = np.array([100, 130, 160])               # component means
    sigma_m = np.array([0.1,0.1,0.1])       # component variances
    sigma_v = np.array([0.8, 0.3, 1])       # component variances
    pi = np.array([0.5, 0.25, 0.25])        # non-uniform mixing weights
    n = len(v)                                  # number of points to generate


    m,v_old = generate_data(mu_m, mu_v, sigma_m, sigma_v, pi, n)     # generated data


    k = mu_m.shape[0] # number of components
    # model = build_model(m, v, k, tau=1./sigma_m**2) # make pymc model object
    model = build_model(m, v, k) # make pymc model object
    mcmc = pymc.MCMC(model) # make an sampler for the model

    # draw the graphical model
    if False:
        pymc.graph.graph(model, format='png', # format='raw'; then: $ dot -Tpng containger.dot > out.png
            prog='dot',
            path=None, name=None,
            consts=True, legend=False,
            collapse_deterministics=False,
            collapse_potentials=False,
            label_edges=True)
        import sys; sys.exit()

    # run the sampler
    niter = 100000
    burn  = 90000
    thin  = 10
    tune_interval = 1000
    mcmc.sample(niter, burn, thin, tune_interval)

    # extract the sampler statistics
    fit = mcmc.stats()
    z_post_mean = fit['Z']['mean']              # posterior component indicators
    mu_m_post_mean = fit['mu_m']['mean']        # posterior component means
    mu_v_post_mean = fit['mu_v']['mean']        # posterior component means
    tau_m_post_mean = fit['tau_m']['mean']      # posterior component precisions
    tau_v_post_mean = fit['tau_v']['mean']      # posterior component precisions
    pi_post_mean = fit['pi']['mean']            # posterior component mixing coefficients
    print "\nPosterior Component Means for mass:", mu_m_post_mean
    print "Posterior Component Means for volume:", mu_v_post_mean
    print "Posterior Component Precisions for mass:", tau_m_post_mean
    print "Posterior Component Precisions for volume:", tau_v_post_mean
    print "Posterior Mixing Coefficients:", np.hstack([pi_post_mean, 1.-sum(pi_post_mean)])

    # pyplot.interactive(True)
    # pymc.Matplot.plot(mcmc)
    # mcmc.summary()

    pyplot.hist(m, bins=50, normed=1)
    plot_normal(mu_m_post_mean, np.sqrt(1./tau_m_post_mean))

    pyplot.figure()
    pyplot.hist(v, bins=50, normed=1)
    plot_normal(mu_v_post_mean, np.sqrt(1./tau_v_post_mean))
    pyplot.show()

    observed_volume = np.array([15.5])

    pi_post_mean_all = np.hstack([pi_post_mean, 1-sum(pi_post_mean)])
    map_component = np.argmax([np.log(pi_post_mean_all[i])+scipy.stats.norm.logpdf(15.5, loc=mu_v_post_mean[i], scale=1./tau_v_post_mean[i]) for i in range(len(mu_v_post_mean))])
    predicted_mass = mu_m_post_mean[map_component]

    print "Given volume observations: ", observed_volume, " predicted: ", predicted_mass

    import ipdb; ipdb.set_trace()

    '''
    Physical parameters for the wheels on a table:

    mass: m
    wheel: x, y, theta, mu_x, mu_y, mu_theta

| mass |    x |    y | theta | mu_x | mu_y | mu_theta | (memo)                      |
|------+------+------+-------+------+------+----------+-----------------------------|
|   27 |    0 |    0 |     0 |    0 |    0 |        0 | unconstrainted table        |
|   27 |  0.9 |  0.3 |     0 |  0.8 |  0.8 |      0.0 | locked wheel in upper-right |
|   27 |  0.9 | -0.3 |     0 |  0.8 |  0.8 |      0.0 | locked wheel in lower-right |
|   27 | -0.9 |  0.3 |     0 |  0.8 |  0.8 |      0.0 | locked wheel in upper-left  |
|   27 | -0.9 | -0.3 |     0 |  0.8 |  0.8 |      0.0 | locked wheel in lower-left  |
|      |      |      |       |      |      |          |                             |
|   10 |    0 |    0 |     0 |    0 |    0 |      0.1 | orange chair                |
|   14 |    0 |    0 |     0 |    0 |    0 |      0.2 | black chair                 |
|      |      |      |       |      |      |          |                             |
|   22 |    0 |    0 |     0 |  0.4 |  0.4 |        0 | brown box                   |
|   17 |    0 |    0 |     0 |  0.3 |  0.3 |        0 | silver box                  |
    '''
