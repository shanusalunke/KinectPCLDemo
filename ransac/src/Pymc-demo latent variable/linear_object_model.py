import numpy as np
import pymc
from matplotlib import pyplot
from matplotlib.mlab import normpdf


'''
Implements a prototype of the generative object model.
In this version we only have two properties: mass and volume.

The training data consist of a bunch of mass and volume
observations, in principle gathered from a robot.
In this demo we generate the data from a linear model with
some output noise.

This model posits a linear relationship between mass and volume:
m = a * v + b + epsilon, epsilon ~ N(0, sigma^2)

Thus our model parameters are a, b, and sigma.  We put gaussian
priors on a and b, and a uniform prior on sigma (since it has
to be positive.)

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


def linear_model(a, b, x, sigma):
    return np.random.normal(a*x + b, sigma)


def quadratic_model(a, b, c, x, sigma):
    return np.random.normal(a*x**2 + b*x + c, sigma)


def generate_data(a, b, v, sigma):
    return linear_model(a, b, v, sigma)


def build_model2(volume_data, mass_data):
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
        # print a*v+b
        print "a",a
        print "v",v
        print "b",b
        return a * v + b

    # weak uniform prior on output noise variance
    sigma = pymc.Uniform('sigma', 0.0, 200.0, value=3.0) # value=10., observed=True

    # overall model likelihood for m ~ a*v + b + noise, noise ~ Normal(0, sigma^2)
    m = pymc.Normal('m', mu=muM, tau=1.0/sigma**2, value=mass_data, observed=True)

    return pymc.Model(locals())

def build_model( multivariate_data, mass_data):
    '''
    :param v: volume (observed data)
    :param m: mass (observed data)
    '''
    # weak gaussian prior on linear regression coefficient
    label = 'a'
    coefficients = []
    for i in range(0,len(v[0])):
        print label,
        coefficients.append(pymc.Normal(label, mu=0, tau=0.1))
        label = chr(ord(label)+1)
        if label == 'z':
            label = 'A'

    # a = pymc.Normal('a', mu=0, tau=0.1)
    # b = pymc.Normal('b', mu=0, tau=0.1)

    # weak gaussian prior on linear regression intercept
    c = pymc.Normal('c', mu=0, tau=0.1)

    @pymc.deterministic
    def muM(a=coefficients, c=c, v=multivariate_data):
        '''
        Implements a linear regression model for mass on volume.
        '''
        retval = c
        for i in range(0,len(v[0])-1):
            print i,
            print "a[i] =", a[i]
            print "v[i] =", v[:,i]
            retval = retval + a[i]*v[:,i]
        print "\n", retval
        return retval

        # return a[0]*v[0][0] + c
        # return a*v + b*ar + c
        #return a*v + b

    # weak uniform prior on output noise variance
    sigma = pymc.Uniform('sigma', 0.0, 200.0, value=3.0) # value=10., observed=True

    # overall model likelihood for m ~ a*v + b + noise, noise ~ Normal(0, sigma^2)
    m = pymc.Normal('m', mu=muM, tau=1.0/sigma**2, value=mass_data, observed=True)
    return pymc.Model(locals())


if __name__ == '__main__':

    v = np.array([[0.0, 0.0, 0.0, 0.887672, 12.527, 45.2769, 37.8034, 3.50499, 0.0, 0.0, 0.0, 0.0, 0.0, 1.53539, 8.10268, 15.3949, 34.6364, 26.6042, 8.16714, 4.82106, 0.712004, 0.0261886, 0.123428, 3.37853, 22.3174, 32.854, 12.6823, 1.9913, 3.03218, 15.8104, 6.46365, 1.29959, 0.0472992], [0.0, 0.0, 0.0, 0.0, 1.8142, 75.8037, 22.3821, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.20211, 41.216, 20.0739, 19.2503, 13.2578, 0.0, 0.0, 0.0, 0.0, 0.0, 3.83999, 2.75477, 28.1541, 25.8983, 14.0535, 19.9512, 2.91603, 2.36043, 0.0677317, 0.00382549], [0.0, 0.0, 0.0, 0.0, 0.0, 21.3456, 66.5776, 12.0486, 0.0281319, 0.0, 0.0, 0.0, 0.101796, 0.939838, 12.4819, 17.892, 34.4132, 26.9439, 6.43302, 0.758212, 0.0361976, 0.0, 0.0167353, 1.94928, 39.5243, 37.8137, 17.7929, 1.81125, 0.66189, 0.223695, 0.206299, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.020652, 36.3163, 49.323, 14.1136, 0.226537, 0.0, 0.0, 0.0, 0.569504, 3.79238, 14.3359, 21.7236, 29.701, 18.3981, 7.63832, 3.35859, 0.482615, 0.0, 0.0232823, 8.32146, 35.4589, 26.4164, 18.9233, 6.30369, 4.14119, 0.314901, 0.0968417, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.69301, 83.4912, 6.81577, 0.0, 0.0, 0.0, 0.0, 0.14339, 0.141157, 0.123805, 1.63297, 6.80354, 57.4656, 32.2591, 1.20897, 0.151633, 0.0, 0.0698545], [0.0, 0.0, 0.0, 0.0, 0.0, 95.197, 4.80297, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.172814, 28.8277, 58.4992, 12.2869, 0.213366, 0.0, 0.0, 0.0, 0.0, 0.0490398, 0.00448481, 1.98569, 41.2433, 30.1197, 25.2127, 0.162212, 1.22298, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 32.666, 67.3341, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00597014, 4.86658, 81.3563, 13.6884, 0.0827378, 0.0, 0.0, 0.0, 0.0191012, 0.057598, 1.01824, 45.8162, 49.673, 3.41579, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 99.9983, 0.0016637, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 99.7715, 0.228471, 0.0, 0.0, 0.0, 0.0, 0.980233, 0.0, 0.00179626, 0.00647536, 2.89593, 95.5836, 0.524786, 0.00436933, 0.00287343, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 49.0464, 50.9536, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.117642, 18.3074, 58.843, 22.6155, 0.116469, 0.0, 0.0, 0.0, 0.0, 0.0, 0.353258, 23.0919, 68.9529, 7.60193, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 33.1548, 66.8452, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00145109, 4.62795, 92.2517, 3.11894, 0.0, 0.0, 0.0, 0.0, 0.0, 0.019507, 0.0305398, 25.4797, 70.3472, 4.1231, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00732534, 0.0, 0.425273, 0.717982, 19.9753, 78.3903, 0.103713, 0.0, 0.0112229, 0.0197575, 0.349195], [0.0, 0.0, 0.0, 0.0, 0.0, 64.1336, 35.8664, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.27338, 21.9851, 59.5296, 17.2119, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0830145, 2.49736, 24.4921, 63.7271, 9.13432, 0.0660921, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 38.5065, 60.534, 0.959527, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0184317, 0.852264, 18.3474, 55.8883, 23.1736, 1.72003, 0.0, 0.0, 0.0, 0.04002, 2.61294, 13.6776, 41.9586, 38.6228, 3.0109, 0.0611952, 0.0, 0.0159085, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 99.9034, 0.0966018, 0.0, 0.0, 0.0, 0.0, 0.0, 3.03857, 0.557832, 6.78539, 36.6, 41.4019, 8.70449, 0.266258, 2.64558, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0768632, 99.9231, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.010511, 0.391168, 0.516687, 3.4938, 95.5472, 0.0346795, 0.00596939, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 79.8292, 20.1708, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.752873, 23.8224, 49.5054, 24.8429, 1.07648, 0.0, 0.0, 0.0, 0.0, 0.905494, 3.85613, 24.7709, 51.1268, 19.1436, 0.132944, 0.0215695, 0.0, 0.0426004, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 78.7734, 21.2266, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.200702, 4.56878, 26.1286, 42.4671, 22.7159, 3.8995, 0.0195156, 0.0, 0.0, 0.232285, 0.91323, 3.28183, 20.6447, 51.0631, 20.6968, 2.69925, 0.402799, 0.0290931, 0.00638971, 0.0304433]])

    # v = np.array([89.489348133314579, 105.03485388021663, 103.82024355732815, 91.504131030847802, 146.64458530427584, 129.37977842755956, 130.42276692736462, 170.58696241511308, 121.06452366220911, 140.44211731437332, 162.92599965310674, 120.31290994548047, 152.22614664286178,  118.43345482451321, 114.78335139061392])
    # m = np.array([1,1,1,2,2,2,2,2,2,2,2,2,3,3])


    m = generate_data(a=2.5, b=2, v=v[:,0], sigma=0.25)

    # v = np.linspace(1, 3, num=20)
    # m = generate_data(a=2.5, b=2, v=v, sigma=0.25)


    model = build_model(v, m)   # make pymc model object
    mcmc = pymc.MCMC(model)     # make an sampler for the model

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
    niter = 10000
    burn  = 9000
    thin  = 10
    tune_interval = 1000
    mcmc.sample(niter, burn, thin, tune_interval)

    # extract the sampler statistics
    fit = mcmc.stats()
    a_post_mean = fit['a']['mean']          # posterior component means
    b_post_mean = fit['b']['mean']          # posterior component precisions
    sigma_post_mean = fit['sigma']['mean']  # posterior component mixing coefficients
    print "\nPosterior linear coefficient:", a_post_mean
    print "Posterior intercept:", b_post_mean
    print "Posterior variance:", sigma_post_mean

    #pyplot.interactive(True)
    #pymc.Matplot.plot(mcmc)
    #mcmc.summary()

    ## Now lets do something with our estimated model!
    # some volume data measured by our kinect:
    observed_volume = np.array([15.5])

    predicted_means = linear_model(
        a=a_post_mean, b=b_post_mean,
        x=observed_volume, sigma=sigma_post_mean)

    print "Given volume observations: ", observed_volume, " predicted: ", predicted_means
    import ipdb; ipdb.set_trace()
