from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
from copy import copy, deepcopy
import sys
import kabuki

def interpolate_trace(x, trace, range=(-1,1), bins=100):
    """Interpolate distribution (from samples) at position x.

    :Arguments:
        x <float>: position at which to evalute posterior.
        trace <np.ndarray>: Trace containing samples from posterior.

    :Optional:
        range <tuple=(-1,1): Bounds of histogram (should be fairly
            close around region of interest).
        bins <int=100>: Bins of histogram (should depend on trace length).

    :Returns:
        float: Posterior density at x.
    """

    import scipy.interpolate

    x_histo = np.linspace(range[0], range[1], bins)
    histo = np.histogram(trace, bins=bins, range=range, normed=True)[0]
    interp = scipy.interpolate.InterpolatedUnivariateSpline(x_histo, histo)(x)

    return interp

def save_csv(data, fname, sep=None):
    """Save record array to fname as csv.

    :Arguments:
        data <np.recarray>: Data array to output.
        fname <str>: File name.

    :Optional:
        sep <str=','>: Separator between columns.

    :SeeAlso: load_csv
    """
    if sep is None:
        sep = ','
    with open(fname, 'w') as fd:
        # Write header
        fd.write(sep.join(data.dtype.names))
        fd.write('\n')
        # Write data
        for line in data:
            line_str = [str(i) for i in line]
            fd.write(sep.join(line_str))
            fd.write('\n')

def load_csv(*args, **kwargs):
    """Load record array from csv.

    :Arguments:
        fname <str>: File name.
        See numpy.recfromcsv

    :Optional:
        See numpy.recfromcsv

    :Note:
        Direct wrapper for numpy.recfromcsv().

    :SeeAlso: save_csv, numpy.recfromcsv
    """
    #read data
    return np.recfromcsv(*args, **kwargs)

def parse_config_file(fname, mcmc=False, load=False, param_names=None):
    """Open, parse and execute a kabuki model as specified by the
    configuration file.

    :Arguments:
        fname <str>: File name of config file.

    :Optional:
        mcmc <bool=False>: Run MCMC on model.
        load <bool=False>: Load from database.
    """
    raise NotImplementedError("Obsolete, todo!")

    import os.path
    if not os.path.isfile(fname):
        raise ValueError("%s could not be found."%fname)

    import ConfigParser

    config = ConfigParser.ConfigParser()
    config.read(fname)

    #####################################################
    # Parse config file
    data_fname = config.get('data', 'load')
    if not os.path.exists(data_fname):
        raise IOError, "Data file %s not found."%data_fname

    try:
        save = config.get('data', 'save')
    except ConfigParser.NoOptionError:
        save = False

    data = np.recfromcsv(data_fname)

    try:
        model_type = config.get('model', 'type')
    except ConfigParser.NoOptionError:
        model_type = 'simple'

    try:
        is_subj_model = config.getboolean('model', 'is_subj_model')
    except ConfigParser.NoOptionError:
        is_subj_model = True

    try:
        no_bias = config.getboolean('model', 'no_bias')
    except ConfigParser.NoOptionError:
        no_bias = True

    try:
        debug = config.getboolean('model', 'debug')
    except ConfigParser.NoOptionError:
        debug = False

    try:
        dbname = config.get('mcmc', 'dbname')
    except ConfigParser.NoOptionError:
        dbname = None

    # Get depends
    depends = {}
    if param_names is not None:
        for param_name in param_names:
            try:
                # Multiple depends can be listed (separated by a comma)
                depends[param_name] = config.get('depends', param_name).split(',')
            except ConfigParser.NoOptionError:
                pass

    # MCMC values
    try:
        samples = config.getint('mcmc', 'samples')
    except ConfigParser.NoOptionError:
        samples = 10000
    try:
        burn = config.getint('mcmc', 'burn')
    except ConfigParser.NoOptionError:
        burn = 5000
    try:
        thin = config.getint('mcmc', 'thin')
    except ConfigParser.NoOptionError:
        thin = 3
    try:
        verbose = config.getint('mcmc', 'verbose')
    except ConfigParser.NoOptionError:
        verbose = 0

    try:
        plot_rt_fit = config.getboolean('stats', 'plot_rt_fit')
    except ConfigParser.NoOptionError, ConfigParser.NoSectionError:
        plot_rt_fit = False

    try:
        plot_posteriors = config.getboolean('stats', 'plot_posteriors')
    except ConfigParser.NoOptionError, ConfigParser.NoSectionError:
        plot_posteriors = False


    print "Creating model..."

    if mcmc:
        if not load:
            print "Sampling... (this can take some time)"
            m.mcmc(samples=samples, burn=burn, thin=thin, verbose=verbose, dbname=dbname)
        else:
            m.mcmc_load_from_db(dbname=dbname)

    if save:
        m.save_stats(save)
    else:
        print m.summary()

    if plot_rt_fit:
        m.plot_rt_fit()

    if plot_posteriors:
        m.plot_posteriors

    return m

def scipy_stochastic(scipy_dist, **kwargs):
    """
    Return a Stochastic subclass made from a particular SciPy distribution.
    """
    import inspect
    import scipy.stats.distributions as sc_dst
    from pymc.ScipyDistributions import separate_shape_args
    from pymc.distributions import new_dist_class, bind_size

    if scipy_dist.__class__.__name__.find('_gen'):
        scipy_dist = scipy_dist(**kwargs)

    name = scipy_dist.__class__.__name__.replace('_gen','').capitalize()

    (args, varargs, varkw, defaults) = inspect.getargspec(scipy_dist._pdf)

    shape_args = args[2:]
    if isinstance(scipy_dist, sc_dst.rv_continuous):
        dtype=float

        def logp(value, **kwds):
            args, zkwds = separate_shape_args(kwds, shape_args)
            if hasattr(scipy_dist, '_logp'):
                return scipy_dist._logp(value, *args)
            else:
                return np.sum(scipy_dist.logpdf(value,*args,**kwds))

        parent_names = shape_args + ['loc', 'scale']
        defaults = [None] * (len(parent_names)-2) + [0., 1.]

    elif isinstance(scipy_dist, sc_dst.rv_discrete):
        dtype=int

        def logp(value, **kwds):
            args, kwds = separate_shape_args(kwds, shape_args)
            if hasattr(scipy_dist, '_logp'):
                return scipy_dist._logp(value, *args)
            else:
                return np.sum(scipy_dist.logpmf(value,*args,**kwds))

        parent_names = shape_args + ['loc']
        defaults = [None] * (len(parent_names)-1) + [0]
    else:
        return None

    parents_default = dict(zip(parent_names, defaults))

    def random(shape=None, **kwds):
        args, kwds = separate_shape_args(kwds, shape_args)

        if shape is None:
            return scipy_dist.rvs(*args, **kwds)
        else:
            return np.reshape(scipy_dist.rvs(*args, **kwds), shape)

    # Build docstring from distribution
    docstr = name[0]+' = '+name + '(name, '+', '.join(parent_names)+', value=None, shape=None, trace=True, rseed=True, doc=None)\n\n'
    docstr += 'Stochastic variable with '+name+' distribution.\nParents are: '+', '.join(parent_names) + '.\n\n'
    docstr += """
Methods:

    random()
        - draws random value
          sets value to return value

    ppf(q)
        - percent point function (inverse of cdf --- percentiles)
          sets value to return value

    isf(q)
        - inverse survival function (inverse of sf)
          sets value to return value

    stats(moments='mv')
        - mean('m',axis=0), variance('v'), skew('s'), and/or kurtosis('k')


Attributes:

    logp
        - sum(log(pdf())) or sum(log(pmf()))

    cdf
        - cumulative distribution function

    sf
        - survival function (1-cdf --- sometimes more accurate)

    entropy
        - (differential) entropy of the RV.


NOTE: If you encounter difficulties with this object, please try the analogous
computation using the rv objects in scipy.stats.distributions directly before
reporting the bug.
    """

    new_class = new_dist_class(dtype, name, parent_names, parents_default, docstr, logp, random, True, None)
    class newer_class(new_class):
        __doc__ = docstr
        rv = scipy_dist
        rv.random = random

        def __init__(self, *args, **kwds):
            new_class.__init__(self, *args, **kwds)
            self.args, self.kwds = separate_shape_args(self.parents, shape_args)
            self.frozen_rv = self.rv(self.args, self.kwds)
            self._random = bind_size(self._random, self.shape)

        def _pymc_dists_to_value(self, args):
            """Replace arguments that are a pymc.Node with their value."""
            # This is needed because the scipy rv function transforms
            # every input argument which causes new pymc lambda
            # functions to be generated. Thus, when calling this many
            # many times, excessive amounts of RAM are used.
            new_args = []
            for arg in args:
                if isinstance(arg, pm.Node):
                    new_args.append(arg.value)
                else:
                    new_args.append(arg)

            return new_args

        def pdf(self, value=None):
            """
            The probability distribution function of self conditional on parents
            evaluated at self's current value
            """
            if value is None:
                value = self.value
            return self.rv.pdf(value, *self._pymc_dists_to_value(self.args), **self.kwds)

        def cdf(self, value=None):
            """
            The cumulative distribution function of self conditional on parents
            evaluated at self's current value
            """
            if value is None:
                value = self.value
            return self.rv.cdf(value, *self._pymc_dists_to_value(self.args), **self.kwds)

        def sf(self, value=None):
            """
            The survival function of self conditional on parents
            evaluated at self's current value
            """
            if value is None:
                value = self.value
            return self.rv.sf(self.value, *self._pymc_dists_to_value(self.args), **self.kwds)

        def ppf(self, q):
            """
            The percentile point function (inverse cdf) of self conditional on parents.
            Self's value will be set to the return value.
            """
            self.value = self.rv.ppf(q, *self._pymc_dists_to_value(self.args), **self.kwds)
            return self.value

        def isf(self, q):
            """
            The inverse survival function of self conditional on parents.
            Self's value will be set to the return value.
            """
            self.value = self.rv.isf(q, *self._pymc_dists_to_value(self.args), **self.kwds)
            return self.value

        def stats(self, moments='mv'):
            """The first few moments of self's distribution conditional on parents"""
            return self.rv.stats(moments=moments, *self._pymc_dists_to_value(self.args), **self.kwds)

        def _entropy(self):
            """The entropy of self's distribution conditional on its parents"""
            return self.rv.entropy(*self._pymc_dists_to_value(self.args), **self.kwds)
        entropy = property(_entropy, doc=_entropy.__doc__)

    newer_class.__name__ = new_class.__name__
    return newer_class

def set_proposal_sd(mc, tau=.1):
    for var in mc.variables:
        if var.__name__.endswith('var'):
            # Change proposal SD
            mc.use_step_method(pm.Metropolis, var, proposal_sd = tau)

    return


###########################################################################
# The following code is directly copied from Twisted:
# http://twistedmatrix.com/trac/browser/tags/releases/twisted-11.1.0/twisted/python/reflect.py
# For the license see:
# http://twistedmatrix.com/trac/browser/trunk/LICENSE
###########################################################################

class _NoModuleFound(Exception):
    """
    No module was found because none exists.
    """

def _importAndCheckStack(importName):
    """
    Import the given name as a module, then walk the stack to determine whether
    the failure was the module not existing, or some code in the module (for
    example a dependent import) failing.  This can be helpful to determine
    whether any actual application code was run.  For example, to distiguish
    administrative error (entering the wrong module name), from programmer
    error (writing buggy code in a module that fails to import).

    @raise Exception: if something bad happens.  This can be any type of
    exception, since nobody knows what loading some arbitrary code might do.

    @raise _NoModuleFound: if no module was found.
    """
    try:
        try:
            return __import__(importName)
        except ImportError:
            excType, excValue, excTraceback = sys.exc_info()
            while excTraceback:
                execName = excTraceback.tb_frame.f_globals["__name__"]
                if (execName is None or # python 2.4+, post-cleanup
                    execName == importName): # python 2.3, no cleanup
                    raise excType, excValue, excTraceback
                excTraceback = excTraceback.tb_next
            raise _NoModuleFound()
    except:
        # Necessary for cleaning up modules in 2.3.
        sys.modules.pop(importName, None)
        raise

def find_object(name):
    """
    Retrieve a Python object by its fully qualified name from the global Python
    module namespace.  The first part of the name, that describes a module,
    will be discovered and imported.  Each subsequent part of the name is
    treated as the name of an attribute of the object specified by all of the
    name which came before it.  For example, the fully-qualified name of this
    object is 'twisted.python.reflect.namedAny'.

    @type name: L{str}
    @param name: The name of the object to return.

    @raise InvalidName: If the name is an empty string, starts or ends with
        a '.', or is otherwise syntactically incorrect.

    @raise ModuleNotFound: If the name is syntactically correct but the
        module it specifies cannot be imported because it does not appear to
        exist.

    @raise ObjectNotFound: If the name is syntactically correct, includes at
        least one '.', but the module it specifies cannot be imported because
        it does not appear to exist.

    @raise AttributeError: If an attribute of an object along the way cannot be
        accessed, or a module along the way is not found.

    @return: the Python object identified by 'name'.
    """

    if not name:
        raise InvalidName('Empty module name')

    names = name.split('.')

    # if the name starts or ends with a '.' or contains '..', the __import__
    # will raise an 'Empty module name' error. This will provide a better error
    # message.
    if '' in names:
        raise InvalidName(
            "name must be a string giving a '.'-separated list of Python "
            "identifiers, not %r" % (name,))

    topLevelPackage = None
    moduleNames = names[:]
    while not topLevelPackage:
        if moduleNames:
            trialname = '.'.join(moduleNames)
            try:
                topLevelPackage = _importAndCheckStack(trialname)
            except _NoModuleFound:
                moduleNames.pop()
        else:
            if len(names) == 1:
                raise ModuleNotFound("No module named %r" % (name,))
            else:
                raise ObjectNotFound('%r does not name an object' % (name,))

    obj = topLevelPackage
    for n in names[1:]:
        obj = getattr(obj, n)

    return obj

######################
# END OF COPIED CODE #
######################

if __name__ == "__main__":
    import doctest
    doctest.testmod()
