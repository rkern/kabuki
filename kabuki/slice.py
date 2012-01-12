import numpy as np
import pymc as pm
rand = np.random.rand

class SliceSampling(pm.StepMethod):
    def __init__(self, stochastic, width, maxiter = 200, low_bound = None, up_bound = None, verbose=None, tally=True,):
        
        pm.StepMethod.__init__(self, [stochastic], tally=tally)
        self.low_bound = low_bound
        self.up_bound = up_bound
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

        assert iter < self.maxiter, "The step-out procedure failed"
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

        assert iter < self.maxiter, "The step-out procedure failed"
        self.neval += iter

        if self.verbose>2:
            print 'after %d iteration interval is [%.3f, %.3f]' % (iter, xl, xr)


        #A new point is found by picking uniformly from the interval [xl, xr].
        xp = rand()*(xr-xl) + xl
        stoch.value = xp

        #shrink the interval (or hyper-rectangle) if a point outside the
        #density is drawn.
        iter = 0
        while(self.logp_plus_loglike < z) and (iter < self.maxiter):
            if (xp > value):
                xr = xp
            else:
                xl = xp
            xp = rand() * (xr-xl) + xl #draw again
            stoch.value = xp
            iter += 1

        assert iter < self.maxiter, "The shrink-in procedure failed."
        self.neval += iter
        if self.verbose>2:
            print 'after %d iteration found new value: %.3f' % (iter, xp)

