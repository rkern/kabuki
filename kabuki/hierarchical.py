 #!/usr/bin/python
from __future__ import division
from copy import copy

import numpy as np
import numpy.lib.recfunctions as rec

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

import pymc as pm
import warnings

import kabuki
from copy import copy, deepcopy
from matplotlib.mlab import rec_drop_fields


class Knode(object):
    def __init__(self, stoch, step_method=None, step_method_args=None, **kwargs):
        self.stoch = stoch
        self.step_method = step_method
        self.args = kwargs
        if step_method_args is None:
            self.step_method_args = {}


class HNode(pm.Node):
    """Specify a parameter of a model.

    :Arguments:
        name <str>: Name of parameter.

    :Optional:
        is_bottom_node <bool=False>: Is node at the bottom of the hierarchy (e.g. likelihoods)

        subj_knode <Knode>: the Knode of the subject

        group_knode <Knode>: the Knode of the group

        var_knode <Knode>: the Knode of the var


        group_label <string>: the label that the subj node gives to the group parameter

        var_label <string>: the label that the subj node gives to the var parameter

        transform <function> : a function which take the group and var nodes and return
            new group and var nodes. for instance if the var node is define as a prior
            over the standard devision and one would like to get a prior on the precision
            then one will define lambda mu,var:(mu, var**-2).

        bottom_stoch

        vars <dict>: User-defined variables, can be anything you later
            want to access.

        optional <bool=False>: Only create distribution when included.
            Otherwise, set to default value (see below).

        default <float>: Default value if optional=True.

        verbose <int=0>: Verbosity.

        var_type <string>: type of the var node, can be one of ['std', 'precision', 'sample_size']
    """

    def __init__(self, name, is_bottom_node=False, vars=None, default=None, optional=False,
                 subj_knode=None, group_knode=None, var_knode=None,
                 group_label=None, var_label=None, var_type=None,
                 transform=None, share_var=False, use_spx=False, verbose=0):

        for (attr, value) in locals().iteritems():
            setattr(self, attr, value)

        self.knodes = {'group': group_knode, 'var': var_knode, 'subj': subj_knode}

        if self.optional and self.default is None:
            raise ValueError("Optional parameters have to have a default value.")

        self.group_nodes = OrderedDict()
        self.var_nodes = OrderedDict()
        self.subj_nodes = OrderedDict()

        # Pointers that get overwritten
        self.group = None
        self.tag = None
        self.data = None
        self.idx = None

    def reset(self):
        self.group = None
        self.tag = None
        self.data = None
        self.idx = None

    def get_full_name(self):
        if self.idx is not None:
            if self.tag is not None:
                return '%s%s%i'%(self.name, self.tag, self.idx)
            else:
                return '%s%i'%(self.name, self.idx)
        else:
            if self.tag is not None:
                return '%s%s'%(self.name, self.tag)
            else:
                return self.name

    full_name = property(get_full_name)

    def __repr__(self):
        return object.__repr__(self).replace(' object ', " '%s' "%self.name)


    def var_func(self, values):
        """ applay the varinace node func (according to var type) on values"""
        if self.var_type == 'std':
            return np.std(values)

        elif self.var_type == 'var':
            return np.std(values) ** 2

        elif self.var_type == 'precision':
            return np.std(values) ** -2

        elif self.var_type == 'sample_size':
            v = np.var(values)
            m = np.mean(values)
            return (m * (1 - m)) / v - 1

        else:
            raise ValueError, "unknown var_type"


class Hierarchical(object):
    """Creation of hierarchical Bayesian models in which each subject
    has a set of parameters that are constrained by a group distribution.

    :Arguments:
        data : numpy.recarray
            Input data with a row for each trial.
            Must contain the following columns:
              * 'rt': Reaction time of trial in seconds.
              * 'response': Binary response (e.g. 0->error, 1->correct)
            May contain:
              * 'subj_idx': A unique ID (int) of the subject.
              * Other user-defined columns that can be used in depends_on
                keyword.

    :Optional:
        include : tuple
            If the model has optional arguments, they
            can be included as a tuple of strings here.

        is_group_model : bool
            If True, this results in a hierarchical
            model with separate parameter distributions for each
            subject. The subject parameter distributions are
            themselves distributed according to a group parameter
            distribution.

        depends_on : dict
            Specifies which parameter depends on data
            of a column in data. For each unique element in that
            column, a separate set of parameter distributions will be
            created and applied. Multiple columns can be specified in
            a sequential container (e.g. list)

            :Example:

            >>> depends_on={'param1':['column1']}

            Suppose column1 has the elements 'element1' and
            'element2', then parameters 'param1('element1',)' and
            'param1('element2',)' will be created and the
            corresponding parameter distribution and data will be
            provided to the user-specified method get_liklihood().

        trace_subjs : bool
             Save trace for subjs (needed for many
             statistics so probably a good idea.)

        plot_var : bool
             Plot group variability parameters
             (i.e. variance of Normal distribution.)

        replace_params : list of Parameters
            User defined parameters to replace the default ones.

        update_params : dictionary that holds dictionaries
            User defined parameters that to update files in the default ones.
            the keys of the dictionary should be the names of the parameters that
            one wants to update. The values are another dictionary with keys for the
            attributes the will be updated to the associated values.
            e.g., to change parameter x's group_step_method to Metropolis
            one should pass the following
            {'x' : {'group_step_method': Metropolis}}


    :Note:
        This class must be inherited. The child class must provide
        the following functions:
            * get_bottom_node(param, params): Return distribution
                  for nodes at the bottom of the hierarchy param (e.g. the model
                  likelihood). params contains the associated model
                  parameters.

        In addition, the variable self.params must be defined as a
        list of Paramater().

    """

    def __init__(self, data, is_group_model=None, depends_on=None, trace_subjs=True,
                 plot_subjs=False, plot_var=False, include=(), replace_params = None,
                 update_params = None):
        # Init
        self.include = set(include)

        self.nodes = {}
        self.mc = None

        # Add data_idx field to data. Since we are restructuring the
        # data, this provides a means of getting the data out of
        # kabuki and sorting to according to data_idx to get the
        # original order.
        self.data = data

        if not depends_on:
            self.depends_on = {}
        else:
            # Support for supplying columns as a single string
            # -> transform to list
            for key in depends_on:
                if type(depends_on[key]) is str:
                    depends_on[key] = [depends_on[key]]
            # Check if column names exist in data
            for depend_on in depends_on.itervalues():
                for elem in depend_on:
                    if elem not in self.data.dtype.names:
                        raise KeyError, "Column named %s not found in data." % elem
            self.depends_on = depends_on
        self.depends_dict = OrderedDict()

        if is_group_model is None:
            if 'subj_idx' in data.dtype.names:
                if len(np.unique(data['subj_idx'])) != 1:
                    self.is_group_model = True
                else:
                    self.is_group_model = False
            else:
                self.is_group_model = False

        else:
            if is_group_model:
                if 'subj_idx' not in data.dtype.names:
                    raise ValueError("Group models require 'subj_idx' column in input data.")

            self.is_group_model = is_group_model

        # Should the model incorporate multiple subjects
        if self.is_group_model:
            self._subjs = np.unique(data['subj_idx'])
            self._num_subjs = self._subjs.shape[0]

        self._init_params(replace_params, update_params, trace_subjs, plot_subjs, plot_var)



    def _init_params(self, replace_params, update_params, trace_subjs, plot_subjs, plot_var):
        """initialize self.params"""

       #set Parameters
        self.params = self.create_params()
        if replace_params == None:
            replace_params = ()

        #change the default parametrs
        self._change_default_params(replace_params, update_params)

        for param in self.params:
            #set has_x_node
            param.has_group_nodes = (param.group_knode is not None)
            param.has_var_nodes = (param.var_knode is not None)
            param.has_subj_nodes = (param.subj_knode is not None)

            #set other attributes
            if param.is_bottom_node:
                param.has_subj_nodes = True
                continue
            # if not trace_subjs and param.has_subj_nodes:
            #     param.subj_knode.args['trace'] = False
            # if not plot_subjs and param.has_subj_nodes:
            #     param.subj_knode.args['plot'] = False
            # if not plot_var and param.has_var_nodes:
            #     param.var_knode.args['plot'] = False

        #set params_dict
        self.params_dict = {}
        for param in self.params:
            self.params_dict[param.name] = param



    def _change_default_params(self, replace_params, update_params):
        """replace/update parameters with user defined parameters"""
        #replace params
        if type(replace_params)==HNode:
            replace_params = [replace_params]
        for new_param in replace_params:
            for i in range(len(self.params)):
                if self.params[i].name == new_param.name:
                    self.params[i] = new_param

        #update params
        if update_params == None:
            return

        for (param_name, dict) in update_params.iteritems():
            found_param = False
            for i in range(len(self.params)):
                if self.params[i].name == param_name:
                    for (key, new_value) in dict.iteritems():
                        if hasattr(self.params[i], key):
                            setattr(self.params[i], key, new_value)
                        else:
                            raise ValueError, "An invalid key (%s) was found in update_params for Parameter %s" % (key, param_name)
                    found_param = True
                    break
            if not found_param:
                raise ValueError, "An invalid parameter (%s) was found in update_params" % (param_name)


    def _get_data_depend(self):
        """Partition data according to self.depends_on.

        :Returns:
            List of tuples with the data, the corresponding parameter
            distribution and the parameter name.

        """

        params = {} # use subj parameters to feed into model
        # Create new params dict and copy over nodes
        for name, param in self.params_include.iteritems():
            # Bottom nodes are created later
            if name in self.depends_on or param.is_bottom_node:
                continue
            if self.is_group_model and (param.has_subj_nodes):
                params[name] = param.subj_nodes['']
            else:
                params[name] = param.group_nodes['']

        depends_on = copy(self.depends_on)

        # Make call to recursive function that does the partitioning
        data_dep = self._get_data_depend_rec(self.data, depends_on, params, [])

        return data_dep

    def _get_data_depend_rec(self, data, depends_on, params, dep_name, param=None):
        """Recursive function to partition data and params according
        to depends_on.

        """
        if len(depends_on) != 0: # If depends are present
            data_params = []
            # Get first param from depends_on
            param_name = depends_on.keys()[0]
            col_name = depends_on.pop(param_name) # Take out param
            depend_elements = np.unique(data[col_name])
            # Loop through unique elements
            for depend_element in depend_elements:
                # Append dependent element name.
                dep_name.append(depend_element)
                # Extract rows containing unique element
                data_dep = data[data[col_name] == depend_element]

                # Add a key that is only the col_name that links to
                # the correct dependent nodes. This is the central
                # trick so that later on the get_bottom_node can use
                # params[col_name] and the bottm node will get linked to
                # the correct nodes automatically.
                param = self.params_include[param_name]

                # Add the node
                if self.is_group_model and param.has_subj_nodes:
                    params[param_name] = param.subj_nodes[str(depend_element)]
                else:
                    params[param_name] = param.group_nodes[str(depend_element)]
                # Recursive call with one less dependency and the selected data.
                data_param = self._get_data_depend_rec(data_dep,
                                                       depends_on=copy(depends_on),
                                                       params=copy(params),
                                                       dep_name = copy(dep_name),
                                                       param = param)
                data_params += data_param
                # Remove last item (otherwise we would always keep
                # adding the dep elems of in one column)
                dep_name.pop()
            return data_params

        else: # Data does not depend on anything (anymore)

            #create_
            if len(dep_name) != 0:
                if len(dep_name) == 1:
                    dep_name_str = str(dep_name[0])
                else:
                    dep_name_str = str(dep_name)
            else:
                dep_name_str = ''

            return [(data, params, dep_name, dep_name_str)]

    def create_nodes(self, max_retries=8):
        """Set group level distributions. One distribution for each
        parameter.

        :Arguments:
            retry : int
                How often to retry when model creation
                failed (due to bad starting values).

        TODO: max_retries is causing bugs, so it was commented out.
            we need to delete the nodes that are created using _create before the next retry.
            I would like to check if we actually need this option

        """
        def _create():
            for name, param in self.params_include.iteritems():
                # Bottom nodes are created elsewhere
                if param.is_bottom_node:
                    continue
                # Check if parameter depends on data
                if name in self.depends_on.keys():
                    self._set_dependent_param(param)
                else:
                    self._set_independet_param(param)

            # Init bottom nodes
            for param in self.params_include.itervalues():
                if not param.is_bottom_node:
                    continue
                self._set_bottom_nodes(param, init=True)

            # Create bottom nodes
            for param in self.params_include.itervalues():
                if not param.is_bottom_node:
                    continue
                self._set_bottom_nodes(param, init=False)

        # Include all defined parameters by default.
        self.non_optional_params = [param.name for param in self.params if not param.optional]

        # Create params dictionary
        self.params_dict = OrderedDict()
        for param in self.params:
            self.params_dict[param.name] = param

        self.params_include = OrderedDict()
        for param in self.params:
            if param.name in self.include or not param.optional:
                self.params_include[param.name] = param

#        for tries in range(max_retries):
#            try:
#                _create()
#            except (pm.ZeroProbability, ValueError) as e:
#                continue
#            break
#        else:
#            print "After %f retries, still not good fit found." %(tries)
#            _create()
        _create()


        # Create model dictionary
        self.nodes = {}
        self.group_nodes = {}
        self.var_nodes = {}
        self.subj_nodes = {}

        for name, param in self.params_include.iteritems():
            for tag, node in param.group_nodes.iteritems():
                self.nodes[name+tag+'_group'] = node
                self.group_nodes[name+tag] = node
            for tag, node in param.subj_nodes.iteritems():
                self.nodes[name+tag+'_subj'] = node
                self.subj_nodes[name+tag] = node
            for tag, node in param.var_nodes.iteritems():
                self.nodes[name+tag+'_var'] = node
                self.var_nodes[name+tag] = node

        #update knodes
        # for name, param in self.params_include.iteritems():
        #     if param.is_bottom_node:
        #         continue
        #     param.knodes['group'].nodes = param.group_nodes
        #     #try to update var knodes and subj knodes if they exist
        #     try:
        #         param.knodes['var'].nodes = param.var_nodes
        #         param.knodes['subj'].nodes = param.subj_nodes
        #     except AttributeError:
        #         pass

        return self.nodes

    def map(self, runs=2, warn_crit=5, method='fmin_powell', **kwargs):
        """
        Find MAP and set optimized values to nodes.

        :Arguments:
            runs : int
                How many runs to make with different starting values
            warn_crit: float
                How far must the two best fitting values be apart in order to print a warning message

        :Returns:
            pymc.MAP object of model.

        :Note:
            Forwards additional keyword arguments to pymc.MAP().

        """

        from operator import attrgetter

        maps = []

        for i in range(runs):
            # (re)create nodes to get new initival values.
            #nodes are not created for the first iteration if they already exist
            if (i > 0) or (not self.nodes):
                self.create_nodes()

            m = pm.MAP(self.nodes)
            m.fit(method, **kwargs)
            print m.logp
            maps.append(m)

        # We want to use values of the best fitting model
        sorted_maps = sorted(maps, key=attrgetter('logp'))
        max_map = sorted_maps[-1]

        # If maximum logp values are not in the same range, there
        # could be a problem with the model.
        if runs >= 2:
            abs_err = np.abs(sorted_maps[-1].logp - sorted_maps[-2].logp)
            if abs_err > warn_crit:
                print "Warning! Two best fitting MAP estimates are %f apart. Consider using more runs to avoid local minima." % abs_err

        # Set values of nodes
        for name, node in max_map._dict_container.iteritems():
            if type(node) is pm.ArrayContainer:
                for i,subj_node in enumerate(node):
                    if not subj_node.observed:
                        self.nodes[name][i].value = subj_node.value
            elif not node.observed:
                self.nodes[name].value = node.value

        return max_map

    def _assign_spx(self, param, loc, scale):
        """assign spx step method to param"""
        self.mc.use_step_method(kabuki.steps.SPXcentered,
                                loc=loc,
                                scale=scale,
                                loc_step_method=param.group_knode.step_method,
                                loc_step_method_args=param.group_knode.step_method_args,
                                scale_step_method=param.var_knode.step_method,
                                scale_step_method_args=param.var_knode.step_method_args,
                                beta_step_method=param.subj_knode.step_method,
                                beta_step_method_args=param.subj_knode.step_method_args)


    def mcmc(self, assign_step_methods = True, *args, **kwargs):
        """
        Returns pymc.MCMC object of model.

        Input:
            assign_step_metheds <bool> : assign the step methods in params to the nodes

            The rest of the arguments are forwards to pymc.MCMC
        """

        if not self.nodes:
            self.create_nodes()

        nodes ={}
        for (name, value) in self.nodes.iteritems():
            if value != None:
                nodes[name] = value

        self.mc = pm.MCMC(nodes, *args, **kwargs)

        if (not assign_step_methods) or (not self.is_group_model):
            return self.mc

        #assign step methods
        for param in self.params:
            #assign SPX when share_var
            if param.use_spx and param.share_var:
                loc = param.group_nodes.values()
                scale = param.var_nodes.values()[0]
                self._assign_spx(param, loc, scale)

            #assign SPX when var is not shared
            elif param.use_spx and not param.share_var:
                for (tag, node) in param.group_nodes.iteritems():
                    loc=param.group_nodes[tag]
                    scale=param.var_nodes[tag]
                    self._assign_spx(param, loc, scale)

            #assign other step methods
            else:
                for knode in param.knodes.itervalues():
                    #check if we need to assign step method
                    if (knode is None) or knode.step_method is None:
                        continue
                    #assign it to all the nodes in knode
                    for node in knode.nodes.itervalues():
                        step = knode.step_method
                        args = knode.step_method_args
                        if node is None:
                            continue
                        #check if it is a single node, otherwise it's an array
                        elif isinstance(node, pm.Stochastic):
                            self.mc.use_step_method(step, node, **args)
                        else:
                            # BUG node_array undefined
                            [self.mc.use_step_method(step, node, **args) for node in node_array]

        return self.mc

    def sample(self, *args, **kwargs):
        """Sample from posterior.

        :Note:
            Forwards arguments to pymc.MCMC.sample().

        """

        #init mc if needed
        if self.mc == None:
            self.mcmc()

        #suppress annoying warnings
        if ('hdf5' in dir(pm.database)) and \
           (type(self.mc.db) is pm.database.hdf5.Database):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', pm.database.hdf5.tables.NaturalNameWarning)

        #sample
        self.mc.sample(*args, **kwargs)

        return self.mc

    def print_group_stats(self, fname=None):
        stats_str = kabuki.analyze.gen_group_stats(self.stats())
        if fname is None:
            print stats_str
        else:
            with open(fname) as fd:
                fd.write(stats_str)

    def print_stats(self, fname=None):
        stats_str = kabuki.analyze.gen_stats(self.stats())
        if fname is None:
            print stats_str
        else:
            with open(fname) as fd:
                fd.write(stats_str)


    def _set_dependent_param(self, param):
        """Set parameter that depends on data.

        :Arguments:
            param_name : string
                Name of parameter that depends on data for
                which to set distributions.

        """

        # Get column names for provided param_name
        depends_on = self.depends_on[param.name]

        # Get unique elements from the columns
        data_dep = self.data[depends_on]
        uniq_data_dep = np.unique(data_dep)

        #update depends_dict
        self.depends_dict[param.name] = uniq_data_dep

        # Loop through unique elements
        for uniq_date in uniq_data_dep:
            # Select data
            data_dep_select = self.data[(data_dep == uniq_date)]

            # Create name for parameter
            tag = str(uniq_date)

            # Create parameter distribution from factory
            param.tag = tag
            param.data = data_dep_select
            param.group_nodes[tag] = self.create_pymc_node(param.group_knode, param.full_name)
            param.reset()

            if self.is_group_model and param.has_subj_nodes:
                # Create appropriate subj parameter
                self._set_subj_nodes(param, tag, data_dep_select)

        return self

    def _set_independet_param(self, param):
        """Set parameter that does _not_ depend on data.

        :Arguments:
            param_name : string
                Name of parameter.

        """

        # Parameter does not depend on data
        # Set group parameter
        param.tag = ''
        param.group_nodes[''] = self.create_pymc_node(param.group_knode, param.full_name)
        param.reset()

        if self.is_group_model and param.has_subj_nodes:
            self._set_subj_nodes(param, '', self.data)

        return self

    def _set_subj_nodes(self, param, tag, data):
        """Set nodes with a parent.

        :Arguments:
            param_name : string
                Name of parameter.
            tag : string
                Element name.
            data : numpy.recarray
                Part of the data the parameter
                depends on.

        """
        # Generate subj variability parameter var

        #Gen variance parameter
        param.data = data
        if param.share_var:
            param.tag = 'var'
        else:
            param.tag = 'var' + tag

        #if param does not share variance or this is the first node created
        # then we create the node
        if (len(param.var_nodes) == 0) or (not param.share_var):
            param.var_nodes[tag] = self.create_pymc_node(param.var_knode, param.full_name)
        else: #we copy the first node
            param.var_nodes[tag] = param.var_nodes.values()[0]

        #Create subj parameters
        param.reset()
        param.tag = tag

        #first set parents
        param.group = param.group_nodes[tag]
        param.var = param.var_nodes[tag]

        if param.transform == None:
            group_node = param.group
            var_node = param.var
        else:
            group_node, var_node = param.transform(param.group, param.var)

        #set subj_stoch_args according to parnets labels
        if param.group_label != None:
            param.subj_knode.args[param.group_label] = group_node
        if param.var_label != None:
            param.subj_knode.args[param.var_label] = var_node

        # Init nodes
        param.subj_nodes[tag] = np.empty(self._num_subjs, dtype=object)
        # now create subj parameter distribution for each subject
        for subj_idx,subj in enumerate(self._subjs):
            data_subj = data[data['subj_idx']==subj]
            param.data = data_subj
            param.idx = subj_idx
            param.subj_nodes[tag][subj_idx] = self.create_pymc_node(param.subj_knode, param.full_name)

        param.reset()
        return self

    def _set_bottom_nodes(self, param, init=False):
        """Set parameter node that has no parent.

        :Arguments:
            param_name : string
                Name of parameter.

        :Optional:
            init : bool
                Initialize parameter.

        """
        # Divide data and parameter distributions according to self.depends_on
        data_dep = self._get_data_depend()

        # Loop through parceled data and params and create an observed stochastic
        for i, (data, params_dep, dep_name_list, dep_name_str) in enumerate(data_dep):
            dep_name = dep_name_str
            if init:
                if self.is_group_model and param.has_subj_nodes:
                    param.subj_nodes[dep_name] = np.empty(self._num_subjs, dtype=object)
                else:
                    param.subj_nodes[dep_name] = None
            else:
                self._create_bottom_node(param, data, params_dep, dep_name, i)

        return self

    def _create_bottom_node(self, param, data, params, dep_name, idx):
        """Create parameter node object which has no parent.

        :Note:
            Called by self._set_bottom_node().

        :Arguments:
            param_name : string
                Name of parameter.
            data : numpy.recarray
                Data on which parameter depends on.
            params : list
                List of parameters the node depends on.
            dep_name : str
                Element name the node depends on.
            idx : int
                Subject index.

        """
        if self.is_group_model:
            for i,subj in enumerate(self._subjs):
                # Select data belonging to subj
                data_subj = data[data['subj_idx'] == subj]
                # Skip if subject was not tested on this condition
                if len(data_subj) == 0:
                    continue
                # Select params belonging to subject
                selected_subj_nodes = {}
                # Create new params dict and copy over nodes
                for selected_param in self.params_include.itervalues():
                    # Since groupless nodes are not created in this function we
                    # have to search for the correct node and include it in
                    # the params.
                    if not selected_param.has_subj_nodes:
                        if selected_param.subj_nodes.has_key(dep_name):
                            selected_subj_nodes[selected_param.name] = selected_param.group_nodes[dep_name]
                        else:
                            selected_subj_nodes[selected_param.name] = params[selected_param.name]
                    else:
                        if selected_param.subj_nodes.has_key(dep_name):
                            selected_subj_nodes[selected_param.name] = selected_param.subj_nodes[dep_name][i]
                        else:
                            selected_subj_nodes[selected_param.name] = params[selected_param.name][i]

                # Call to the user-defined function!
                param.tag = dep_name
                param.idx = i
                param.data = data_subj
                param.subj_nodes[dep_name][i] = self.get_bottom_node(param, selected_subj_nodes)
                param.reset()
        else: # Do not use subj params, but group ones
            # Since group nodes are not created in this function we
            # have to search for the correct node and include it in
            # the params
            for selected_param in self.params_include.itervalues():
                if selected_param.subj_nodes.has_key(dep_name):
                    params[selected_param.name] = selected_param.subj_nodes[dep_name]

            param.tag = dep_name
            param.data = data
            param.subj_nodes[dep_name] = self.get_bottom_node(param, params)
            param.reset()

        return self


    def get_node(self, node_name, params):
        """Returns the node object with node_name from params if node
        is included in model, otherwise returns default value.

        """
        if node_name in self.include:
            return params[node_name]
        else:
            assert self.params_dict[node_name].default is not None, "Default value of not-included parameter not set."
            return self.params_dict[node_name].default


    def stats(self, *args, **kwargs):
        """
        smart call of MCMC.stats() for the model
        """
        try:
            nchains = self.mc.db.chains
        except AttributeError:
            raise ValueError("No model found.")

        #check which chain is going to be "stat"
        if 'chain' in kwargs:
            i_chain = kwargs['chain']
        else:
            i_chain = nchains

        #compute stats
        try:
            if self._stats_chain==i_chain:
                return self._stats
        except AttributeError:
            pass
        self._stats = self.mc.stats(*args, **kwargs)
        self._stats_chain = i_chain
        return self._stats



    def load_db(self, dbname, verbose=0, db_loader=None):
        """Load samples from a database created by an earlier model
        run (e.g. by calling .mcmc(dbname='test'))
        """
        if db_loader is None:
            db_loader = pm.database.sqlite.load

        # Set up model
        if not self.nodes:
            self.create_nodes()

        # Open database
        db = db_loader(dbname)

        # Create mcmc instance reading from the opened database
        self.mc = pm.MCMC(self.nodes, db=db, verbose=verbose)

        # Not sure if this does anything useful, but calling for good luck
        self.mc.restore_sampler_state()

        # Take the traces from the database and feed them into our
        # distribution variables (needed for _gen_stats())

        return self


    def create_pymc_node(self, knode, name):
        """Create pymc node named 'name' out of knode """

        if knode is None:
            return None
        else:
            return knode.__class__(name, **knode.parents)

    def init_from_existing_model(self, pre_model, step_method, **kwargs):
        """
        initialize the value and step methods of the model using an existing model
        """
        if not self.nodes:
            self.mcmc(**kwargs)
        all_nodes = list(self.mc.stochastics)
        all_pre_nodes = list(pre_model.mc.stochastics)
        assigned_values = 0
        assigned_steps = 0

        #loop over all nodes
        for i_node in range(len(all_nodes)):
            #get name of current node
            t_name = all_nodes[i_node].__name__
            #get the matched node from the pre_model
            pre_node = [x for x in all_pre_nodes if x.__name__ == t_name]
            if len(pre_node)==0:
                continue
            pre_node = pre_node[0]
            assigned_values += 1

            all_nodes[i_node].value= pre_node.value
            if step_method:
                step_pre_node = pre_model.mc.step_method_dict[pre_node][0]
                pre_sd = step_pre_node.proposal_sd * step_pre_node.adaptive_scale_factor
                if type(step_pre_node) == pm.Metropolis:
                    self.mc.use_step_method(pm.Metropolis(all_nodes[i_node],
                                                          proposal_sd = pre_sd))
                    assigned_steps += 1

        print "assigned values to %d nodes (out of %d)." % (assigned_values, len(all_nodes))
        if step_method:
            print "assigned step methods to %d (out of %d)." % (assigned_steps, len(all_nodes))

    def plot_posteriors(self, parameters=None, plot_subjs=False):
        """
        plot the nodes posteriors
        Input:
            parameters (optional) - a list of parameters to plot.
            plot_subj (optional) - plot subjs nodes

        TODO: add attributes plot_subjs and plot_var to kabuki
        which will change the plot attribute in the relevant nodes
        """

        if parameters is None: #plot the model
            pm.Matplot.plot(self.mc)

        else: #plot only the given parameters

            if type(parameters) != list:
                 parameters = [parameters]

            #get the nodes which will be plotted
            for param in parameters:
                nodes = np.unique(param.group_nodes.values() + param.var_nodes.values())
                if plot_subjs:
                    for nodes_array in param.subj_nodes.values():
                        nodes += list(nodes_array)
            #this part does the ploting
            for node in nodes:
                plot_value = node.plot
                node.plot = True
                pm.Matplot.plot(node)
                node.plot = plot_value

    def subj_by_subj_map_init(self, runs=2, **map_kwargs):
        """
        initializing nodes by finding the MAP for each subject separately
        Input:
            runs - number of MAP runs for each subject
            map_kwargs - other arguments that will be passes on to the map function

        Note: This function should be run prior to the nodes creation, i.e.
        before running mcmc() or map()
        """

        #check if nodes were created. if they were it cause problems for deepcopy
        assert (not self.nodes), "function should be used before nodes are initialized."

        #init
        subjless = {}
        subjs = self._subjs
        n_subjs = len(subjs)
        empty_s_model = deepcopy(self)
        empty_s_model.is_group_model = False
        del empty_s_model._num_subjs, empty_s_model._subjs, empty_s_model.data

        self.create_nodes()

        #fit and copy single subjects
        for i_subj in range(n_subjs):
            #create and fit a single subject
            print "*!*!* fitting subject %d *!*!*" % subjs[i_subj]
            t_data = self.data[self.data['subj_idx'] == subjs[i_subj]]
            t_data = rec_drop_fields(t_data, ['data_idx'])
            s_model = deepcopy(empty_s_model)
            s_model.data = t_data
            s_model.map(method='fmin_powell', runs=runs, **map_kwargs)

            # copy to original model
            for (name, node) in s_model.group_nodes.iteritems():
                #try to assign the value of the node to the original model
                try:
                    self.subj_nodes[name][i_subj].value = node.value
                #if it fails it mean the param has no subj nodes
                except KeyError:
                    if subjless.has_key(name):
                        subjless[name].append(node.value)
                    else:
                        subjless[name] = [node.value]

        #set group and var nodes for params with subjs
        for (param_name, param) in self.params_dict.iteritems():
            #if param has subj nodes than compute group and var nodes from them
            if param.has_subj_nodes:
                for (tag, nodes) in param.subj_nodes.iteritems():
                    subj_values = [x.value for x in nodes]
                    #set group node
                    if param.has_group_nodes:
                        param.group_nodes[tag].value = np.mean(subj_values)
                    #set var node
                    if param.has_var_nodes:
                        param.var_nodes[tag].value = param.var_func(subj_values)

        #set group nodes of subjless nodes
        for (name, values) in subjless.iteritems():
            self.group_nodes[name].value = np.mean(subjless[name])
