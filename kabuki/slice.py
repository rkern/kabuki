import numpy as np
import pymc as pm
rand = np.random.rand

class SliceSampling(pm.StepMethod):
    """
    simple slice sampler
    """
    def __init__(self, stochastic, width = 0.5, maxiter = 200, verbose=None, tally=True,):
        """
        Input:
            stochistic - stochastic node
            width - the initial width of the interval
            maxiter - maximum number of iteration allowed for stepping-out and shrinking
        """
        pm.StepMethod.__init__(self, [stochastic], tally=tally)
        self.width = width
        self.neval = 0
        self.maxiter = maxiter
        self.stochastic = stochastic
        if verbose != None:
            self.verbose = verbose
        else:
            self.verbose = stochastic.verbose

    def step(self):
        stoch = self.stochastic
        value = stoch.value

        #sample vertical level
        z = self.logp_plus_loglike - np.random.exponential()

        if self.verbose>2:
            print self._id + ' current value: %.3f' % value
            print self._id + ' sampled vertical level ' + `z`

        #position an interval at random starting position around the current value
        r = self.width * np.random.rand()
        xl = value - r
        xr = xl + self.width

        if self.verbose>2:
            print 'initial interval [%.3f, %.3f]' % (xl, xr)

        #step out to the left
        iter = 0
        stoch.value = xl
        while (self.logp_plus_loglike > z) and (iter < self.maxiter):
            xl -= self.width
            stoch.value = xl
            iter += 1

        assert iter < self.maxiter, "Step-out procedure failed"
        self.neval += iter
        if self.verbose>2:
            print 'after %d iteration interval is [%.3f, %.3f]' % (iter, xl, xr)

        #step out to the right
        iter = 0
        stoch.value = xr
        while (self.logp_plus_loglike > z) and (iter < self.maxiter):
            xr += self.width
            stoch.value = xr
            iter += 1

        assert iter < self.maxiter, "Step-out procedure failed"
        self.neval += iter
        if self.verbose>2:
            print 'after %d iteration interval is [%.3f, %.3f]' % (iter, xl, xr)

        #draw a new point from the interval [xl, xr].
        xp = rand()*(xr-xl) + xl
        stoch.value = xp

        #if the point is outside the interval than shrink it and draw again
        iter = 0
        while(self.logp_plus_loglike < z) and (iter < self.maxiter):
            if (xp > value):
                xr = xp
            else:
                xl = xp
            xp = rand() * (xr-xl) + xl #draw again
            stoch.value = xp
            iter += 1

        assert iter < self.maxiter, "Shrink-in procedure failed."
        self.neval += iter
        if self.verbose>2:
            print 'after %d iteration found new value: %.3f' % (iter, xp)

