#!/usr/bin/env python
#
# Example usage:
#
# NOTE:
# optype=real: Gaussian parametrization uses a non-linear transformation of split times
#   variance should decrease as split_time increases:
#   variance at node j = variance_coef * (sigmoid(sigmoid_coef * t_j) - sigmoid(sigmoid_coef * t_{parent(j)}))
#   non-linear transformation should be a monotonically non-decreasing function
#   sigmoid has a saturation effect: children will be similar to parent as we go down the tree
#   split times t_j scales inversely with the number of dimensions

#import ipdb
import sys
import os
import optparse
import math
import time
import cPickle as pickle
import random
import pprint as pp
import numpy as np
from warnings import warn
#from utils import hist_count, logsumexp, softmax, sample_multinomial, \
#        sample_multinomial_scores, empty, assert_no_nan, check_if_zero, check_if_one, \
#        multiply_gaussians, divide_gaussians, sigmoid, logsumexp_array
#from mondrianforest_utils import Forest, Param, parser_add_common_options, parser_check_common_options, \
#        bootstrap, parser_add_mf_options, parser_check_mf_options, reset_random_seed, \
#        load_data, add_stuff_2_settings, compute_gaussian_pdf, compute_gaussian_logpdf, \
#        get_filename_mf, precompute_minimal, compute_left_right_statistics, \
#        create_prediction_tree, init_prediction_tree, update_predictive_posterior_node, \
#        compute_metrics_classification, compute_metrics_regression, \
#        update_posterior_node_incremental, init_update_posterior_node_incremental
from itertools import izip, count, chain
from collections import defaultdict

import scipy.io
from warnings import warn
try:
    from scipy.special import gammaln, digamma
    #from scipy.special import gdtrc         # required only for regression
    #from scipy.optimize import fsolve       # required only for regression
    import scipy.stats
    from scipy.stats.stats import pearsonr
except:
    print 'Error loading scipy modules; might cause error later'
from copy import copy
try:
    from sklearn import feature_selection
except:
    print 'Error loading sklearn; might cause error later'


# setting numpy options to debug RuntimeWarnings
#np.seterr(divide='raise')
np.seterr(divide='ignore')      # to avoid warnings for np.log(0)
np.seterr(invalid='ignore')      # to avoid warnings for inf * 0 = nan
np.set_printoptions(precision=3)
np.set_printoptions(linewidth=200)
# color scheme for mondrian
# colors_list = ['DarkRed', 'Navy', 'DimGray', 'Beige']
# other nice colors: Beige, MediumBlue, DarkRed vs FireBrick
colors_list = ['LightGray']  # paused leaf will always be shaded gray
LW = 2
FIGSIZE = (12, 9)
INF = np.inf


def print_forest_stats(mf, settings, data):
    tree_stats = np.zeros((settings.n_mondrians, 2))
    tree_average_depth = np.zeros(settings.n_mondrians)
    for i_t, tree in enumerate(mf.forest):
        tree_stats[i_t, -2:] = np.array([len(tree.leaf_nodes), len(tree.non_leaf_nodes)])
        tree_average_depth[i_t] = tree.get_average_depth(settings, data)[0]
    print 'mean(num_leaves) = %.1f, mean(num_non_leaves) = %.1f, mean(tree_average_depth) = %.1f' \
            % (np.mean(tree_stats[:, -2]), np.mean(tree_stats[:, -1]), np.mean(tree_average_depth))
    print 'n_train = %d, log_2(n_train) = %.1f, mean(tree_average_depth) = %.1f +- %.1f' \
            % (data['n_train'], np.log2(data['n_train']), np.mean(tree_average_depth), np.std(tree_average_depth))

########################
def compute_kl(mean_1, log_prec_1, mean_2, log_prec_2):
    """
    compute KL between two Gaussian distributions
    """
    mean_1 = np.asarray(mean_1)
    log_prec_1 = np.asarray(log_prec_1)
    mean_2 = np.asarray(mean_2)
    log_prec_2 = np.asarray(log_prec_2)
    len_1 = len(mean_1)
    len_2 = len(mean_2)
    len_max = max(len_1, len_2)
    var_1, var_2 = np.exp(-log_prec_1), np.exp(-log_prec_2)     # computationally more stable?
    kl = 0.5 * ( log_prec_1 - log_prec_2 - 1 + np.exp(log_prec_2 - log_prec_1) + np.power(mean_1-mean_2,2)/var_2 )
    # when log_prec_2 > -np.inf, log_prec_1=-np.inf leads to infinite kl
    if len_1 == 1:
        cond = np.logical_and(np.isinf(log_prec_1), log_prec_1 < 0)
        idx_log_prec_1_neginf = np.ones(len_max) * cond
        if cond:
            kl = np.inf * np.ones(len_max)
    else:
        idx_log_prec_1_neginf = np.logical_and(np.isinf(log_prec_1), log_prec_1 < 0)
        kl[idx_log_prec_1_neginf] = np.inf
    if len_2 == 1:
        cond = np.logical_and(np.isinf(log_prec_2), log_prec_2 < 0)
        idx_log_prec_2_neginf = np.ones(len_max) * cond
    else:
        idx_log_prec_2_neginf = np.logical_and(np.isinf(log_prec_2), log_prec_2 < 0)
    # when log_prec_2 = -np.inf, log_prec_1=-np.inf leads to zero kl
    idx_both_log_prec_neginf = np.logical_and(idx_log_prec_1_neginf, idx_log_prec_2_neginf)
    kl[idx_both_log_prec_neginf] = 0.
    # when log_prec_2 = np.inf, any value of log_prec_1 leads to infinite kl
    idx_log_prec_2_posinf = np.logical_and(np.isinf(log_prec_2), log_prec_2 > 0)
    if (len_2 == 1) and idx_log_prec_2_posinf:
        kl = np.inf * np.ones(len_max)
    else:
        kl[idx_log_prec_2_posinf] = np.inf
    if False:
        print 'log_prec_1 = %s, log_prec_2 = %s, kl = %s' % (log_prec_1, log_prec_2, kl)
    if np.any(np.isnan(kl)):
        print '\nsomething went wrong with kl computation'
        print 'var_1 = %s, var_2 = %s' % (var_1, var_2)
        print 'log_prec_1 = %s, log_prec_2 = %s' % (log_prec_1, log_prec_2)
        print 'idx_log_prec_1_neginf = %s' % idx_log_prec_1_neginf
        print 'idx_log_prec_2_neginf = %s' % idx_log_prec_2_neginf
        print 'idx_log_prec_2_posinf = %s' % idx_log_prec_2_posinf
        print 'kl = %s' % kl
        raise Exception
    return kl


def test_compute_kl():
    compute_kl(0*np.ones(2), np.inf*np.ones(2), 0*np.ones(1), np.inf*np.ones(1))
    compute_kl(0*np.ones(2), -np.inf*np.ones(2), 0*np.ones(1), np.inf*np.ones(1))
    compute_kl(0*np.ones(2), np.inf*np.ones(2), 0*np.ones(1), -np.inf*np.ones(1))
    compute_kl(0*np.ones(2), -np.inf*np.ones(2), 0*np.ones(1), -np.inf*np.ones(1))
    compute_kl(0*np.ones(1), np.inf*np.ones(1), 0*np.ones(2), np.inf*np.ones(2))
    compute_kl(0*np.ones(1), -np.inf*np.ones(1), 0*np.ones(2), np.inf*np.ones(2))
    compute_kl(0*np.ones(1), np.inf*np.ones(1), 0*np.ones(2), -np.inf*np.ones(2))
    compute_kl(0*np.ones(1), -np.inf*np.ones(1), 0*np.ones(2), -np.inf*np.ones(2))


def multiply_gaussians(*params):
    """
    input is a list containing (variable number of) gaussian parameters
    each element is a numpy array containing mean and precision of that gaussian
    """
    precision_op, mean_op = 0., 0.
    for param in params:
        precision_op += param[1]
        mean_op += param[0] * param[1]
    mean_op /= precision_op
    return np.array([mean_op, precision_op])


def divide_gaussians(mean_precision_num, mean_precision_den):
    """
    mean_precision_num are parameters of gaussian in the numerator
    mean_precision_den are parameters of gaussian in the denominator
    output is a valid gaussian only if the variance of ouput is non-negative
    """
    precision_op = mean_precision_num[1] - mean_precision_den[1]
    try:
        assert precision_op >= 0.        #   making it > so that mean_op is not inf
    except AssertionError:
        print 'inputs = %s, %s' % (mean_precision_num, mean_precision_den)
        print 'precision_op = %s' % (precision_op)
        raise AssertionError
    if precision_op == 0.:
        mean_op = 0.
    else:
        mean_op = (mean_precision_num[0] * mean_precision_num[1] \
                     - mean_precision_den[0] * mean_precision_den[1] ) / precision_op
    return np.array([mean_op, precision_op])


def hist_count(x, basis):
    """
    counts number of times each element in basis appears in x
    op is a vector of same size as basis
    assume no duplicates in basis
    """
    op = np.zeros((len(basis)), dtype=int)
    map_basis = {}
    for n, k in enumerate(basis):
        map_basis[k] = n
    for t in x:
        op[map_basis[t]] += 1
    return op


def logsumexp(x):
    tmp = x.copy()
    tmp_max = np.max(tmp)
    tmp -= tmp_max
    op = np.log(np.sum(np.exp(tmp))) + tmp_max
    return op


def logsumexp_array(v1, v2):
    """
    computes logsumexp of each element in v1 and v2
    """
    v_min = np.minimum(v1, v2)
    v_max = np.maximum(v1, v2)
    op = v_max + np.log(1 + np.exp(v_min - v_max))
    return op


def logsumexp_2(x, y):
    # fast logsumexp for 2 variables
    # output = log (e^x + e^y) = log(e^max(1+e^(min-max))) = max + log(1 + e^(min-max))
    if x > y:
        min_val = y
        max_val = x
    else:
        min_val = x
        max_val = y
    op = max_val + math.log(1 + math.exp(min_val - max_val))
    return op


def softmax(x):
    tmp = x.copy()
    tmp_max = np.max(tmp)
    tmp -= float(tmp_max)
    tmp = np.exp(tmp)
    op = tmp / np.sum(tmp)
    return op


def assert_no_nan(mat, name='matrix'):
    try:
        assert(not any(np.isnan(mat)))
    except AssertionError:
        print '%s contains NaN' % name
        print mat
        raise AssertionError

def check_if_one(val):
    try:
        assert(np.abs(val - 1) < 1e-9)
    except AssertionError:
        print 'val = %s (needs to be equal to 1)' % val
        raise AssertionError

def check_if_zero(val):
    try:
        assert(np.abs(val) < 1e-9)
    except AssertionError:
        print 'val = %s (needs to be equal to 0)' % val
        raise AssertionError


def linear_regression(x, y):
    ls = np.linalg.lstsq(x, y)
    #print ls
    coef = ls[0]
    if ls[1]:
        sum_squared_residuals = float(ls[1])    # sum of squared residuals
    else:
        sum_squared_residuals = np.sum(np.dot(x, coef) - y)    # sum of squared residuals
    return (coef, sum_squared_residuals)


def sample_multinomial(prob):
    try:
        k = int(np.where(np.random.multinomial(1, prob, size=1)[0]==1)[0])
    except TypeError:
        print 'problem in sample_multinomial: prob = '
        print prob
        raise TypeError
    except:
        raise Exception
    return k


def sample_multinomial_scores(scores):
    scores_cumsum = np.cumsum(scores)
    s = scores_cumsum[-1] * np.random.rand(1)
    k = int(np.sum(s > scores_cumsum))
    return k


def sample_multinomial_scores_old(scores):
    scores_cumsum = np.cumsum(scores)
    s = scores_cumsum[-1] * np.random.rand(1)
    k = 0
    while s > scores_cumsum[k]:
        k += 1
    return k


def sample_polya(alpha_vec, n):
    """ alpha_vec is the parameter of the Dirichlet distribution, n is the #samples """
    prob = np.random.dirichlet(alpha_vec)
    n_vec = np.random.multinomial(n, prob)
    return n_vec


def get_kth_minimum(x, k=1):
    """ gets the k^th minimum element of the list x
        (note: k=1 is the minimum, k=2 is 2nd minimum) ...
        based on the incomplete selection sort pseudocode """
    n = len(x)
    for i in range(n):
        minIndex = i
        minValue = x[i]
        for j in range(i+1, n):
            if x[j] < minValue:
                minIndex = j
                minValue = x[j]
        x[i], x[minIndex] = x[minIndex], x[i]
    return x[k-1]


class empty(object):
    def __init__(self):
        pass


def sigmoid(x):
    op = 1.0 / (1 + np.exp(-x))
    return op


def compute_m_sd(x):
    m = np.mean(x)
    s = np.sqrt(np.var(x))
    return (m, s)

########################
class Forest(object):
    def __init__(self):
        pass

    def predict(self, data, x, settings, param, weights):
        if settings.optype == 'class':
            pred_forest = {'pred_prob': np.zeros((x.shape[0], data['n_class']))}
        else:
            pred_forest = {'pred_mean': np.zeros(x.shape[0]), 'pred_var': np.zeros(x.shape[0]), \
                            'pred_prob': np.zeros(x.shape[0]), 'log_pred_prob': -np.inf*np.ones(x.shape[0]), \
                            'pred_sample': np.zeros(x.shape[0])}
        if settings.debug:
            check_if_one(weights.sum())
        if settings.verbose >= 2:
            print 'weights = \n%s' % weights
        for i_t, tree in enumerate(self.forest):
            pred_all = predictions_tree(tree, x, data, param, settings)
            if settings.optype == 'class':
                # doesn't make sense to average predictive probabilities across trees for real outputs
                pred_prob = pred_all['pred_prob']
                pred_forest['pred_prob'] += weights[i_t] * pred_prob
            #elif settings.optype == 'real':
            #    # skipping pred_prob for real outputs
            #    pred_forest['pred_mean'] += weights[i_t] * pred_all['pred_mean']
            #    pred_forest['pred_var'] += weights[i_t] * pred_all['pred_second_moment']
            #    pred_forest['pred_sample'] += weights[i_t] * pred_all['pred_sample']
            #    pred_forest['log_pred_prob'] = logsumexp_array(pred_forest['log_pred_prob'], \
            #            np.log(weights[i_t]) + pred_all['log_pred_prob'])
        #if settings.optype == 'real':
        #    pred_forest['pred_var'] -= pred_forest['pred_mean'] ** 2
        #    pred_forest['pred_prob'] = np.exp(pred_forest['log_pred_prob'])
        #    # NOTE: logpdf takes in variance
        #    log_prob2 = compute_gaussian_logpdf(pred_forest['pred_mean'], pred_forest['pred_var'], y)
        #    if settings.verbose >= 1:
        #        print 'log_prob (using Gaussian approximation) = %f' % np.mean(log_prob2)
        #        print 'log_prob (using mixture of Gaussians) = %f' % np.mean(pred_forest['log_pred_prob'])
        #    try:
        #        assert np.all(pred_forest['pred_prob'] > 0.)
        #    except AssertionError:
        #        print 'pred prob not > 0'
        #        print 'min value = %s' % np.min(pred_forest['pred_prob'])
        #        print 'sorted array = %s' % np.sort(pred_forest['pred_prob'])
        #        # raise AssertionError
        if settings.debug and settings.optype == 'class':
            check_if_zero(np.mean(np.sum(pred_forest['pred_prob'], axis=1) - 1))
        return pred_forest

    def evaluate_predictions(self, data, x, y, settings, param, weights, print_results=True):
        if settings.optype == 'class':
            pred_forest = {'pred_prob': np.zeros((x.shape[0], data['n_class']))}
        else:
            pred_forest = {'pred_mean': np.zeros(x.shape[0]), 'pred_var': np.zeros(x.shape[0]), \
                            'pred_prob': np.zeros(x.shape[0]), 'log_pred_prob': -np.inf*np.ones(x.shape[0]), \
                            'pred_sample': np.zeros(x.shape[0])}
        if settings.debug:
            check_if_one(weights.sum())
        if settings.verbose >= 2:
            print 'weights = \n%s' % weights
        for i_t, tree in enumerate(self.forest):
            pred_all = evaluate_predictions_tree(tree, x, y, data, param, settings)
            if settings.optype == 'class':
                # doesn't make sense to average predictive probabilities across trees for real outputs
                pred_prob = pred_all['pred_prob']
                pred_forest['pred_prob'] += weights[i_t] * pred_prob
            elif settings.optype == 'real':
                # skipping pred_prob for real outputs
                pred_forest['pred_mean'] += weights[i_t] * pred_all['pred_mean']
                pred_forest['pred_var'] += weights[i_t] * pred_all['pred_second_moment']
                pred_forest['pred_sample'] += weights[i_t] * pred_all['pred_sample']
                pred_forest['log_pred_prob'] = logsumexp_array(pred_forest['log_pred_prob'], \
                        np.log(weights[i_t]) + pred_all['log_pred_prob'])
        if settings.optype == 'real':
            pred_forest['pred_var'] -= pred_forest['pred_mean'] ** 2
            pred_forest['pred_prob'] = np.exp(pred_forest['log_pred_prob'])
            # NOTE: logpdf takes in variance
            log_prob2 = compute_gaussian_logpdf(pred_forest['pred_mean'], pred_forest['pred_var'], y)
            if settings.verbose >= 1:
                print 'log_prob (using Gaussian approximation) = %f' % np.mean(log_prob2)
                print 'log_prob (using mixture of Gaussians) = %f' % np.mean(pred_forest['log_pred_prob'])
            try:
                assert np.all(pred_forest['pred_prob'] > 0.)
            except AssertionError:
                print 'pred prob not > 0'
                print 'min value = %s' % np.min(pred_forest['pred_prob'])
                print 'sorted array = %s' % np.sort(pred_forest['pred_prob'])
                # raise AssertionError
        if settings.debug and settings.optype == 'class':
            check_if_zero(np.mean(np.sum(pred_forest['pred_prob'], axis=1) - 1))
        if settings.optype == 'class':
            # True ignores log prob computation
            metrics = compute_metrics_classification(y, pred_forest['pred_prob'], True)
        else:
            metrics = compute_metrics_regression(y, pred_forest['pred_mean'], pred_forest['pred_prob'])
            if settings.optype == 'real':
                metrics['log_prob2'] = log_prob2
        if print_results:
            if settings.optype == 'class':
                print 'Averaging over all trees, accuracy = %f' % metrics['acc']
            else:
                print 'Averaging over all trees, mse = %f, rmse = %f, log_prob = %f' % (metrics['mse'], \
                        math.sqrt(metrics['mse']), metrics['log_prob'])
        return (pred_forest, metrics)

def predictions_tree(tree, x, data, param, settings):
    if settings.optype == 'class':
        pred_prob = tree.predict_class(x, data['n_class'], param, settings)
        pred_all = {'pred_prob': pred_prob}
    return pred_all

def evaluate_predictions_tree(tree, x, y, data, param, settings):
    if settings.optype == 'class':
        pred_prob = tree.predict_class(x, data['n_class'], param, settings)
        pred_all = {'pred_prob': pred_prob}
    else:
        pred_mean, pred_var, pred_second_moment, log_pred_prob, pred_sample = \
                tree.predict_real(x, y, param, settings)
        pred_all =  {'log_pred_prob': log_pred_prob, 'pred_mean': pred_mean, \
                        'pred_second_moment': pred_second_moment, 'pred_var': pred_var, \
                        'pred_sample': pred_sample}
    return pred_all


def compute_gaussian_pdf(e_x, e_x2, x):
    variance = np.maximum(0, e_x2 - e_x ** 2)
    sd = np.sqrt(variance)
    z = (x - e_x) / sd
    # pdf = np.exp(-(z**2) / 2.) / np.sqrt(2*math.pi) / sd
    log_pdf = -0.5*(z**2) -np.log(sd) -0.5*np.log(2*math.pi)
    pdf = np.exp(log_pdf)
    return pdf


def compute_gaussian_logpdf(e_x, variance, x):
    assert np.all(variance > 0)
    sd = np.sqrt(variance)
    z = (x - e_x) / sd
    log_pdf = -0.5*(z**2) -np.log(sd) -0.5*np.log(2*math.pi)
    return log_pdf


def parser_add_common_options():
    parser = optparse.OptionParser()
    parser.add_option('--dataset', dest='dataset', default='toy-mf',
            help='name of the dataset  [default: %default]')
    parser.add_option('--normalize_features', dest='normalize_features', default=1, type='int',
            help='do you want to normalize features in 0-1 range? (0=False, 1=True) [default: %default]')
    parser.add_option('--select_features', dest='select_features', default=0, type='int',
            help='do you wish to apply feature selection? (1=True, 0=False) [default: %default]')
    parser.add_option('--optype', dest='optype', default='class',
            help='nature of outputs in your dataset (class/real) '\
            'for (classification/regression)  [default: %default]')
    parser.add_option('--data_path', dest='data_path', default='../../process_data/',
            help='path of the dataset [default: %default]')
    parser.add_option('--debug', dest='debug', default='0', type='int',
            help='debug or not? (0=False, 1=everything, 2=special stuff only) [default: %default]')
    parser.add_option('--op_dir', dest='op_dir', default='results',
            help='output directory for pickle files (NOTE: make sure directory exists) [default: %default]')
    parser.add_option('--tag', dest='tag', default='',
            help='additional tag to identify results from a particular run [default: %default]' \
                    ' tag=donottest reduces test time drastically (useful for profiling training time)')
    parser.add_option('--save', dest='save', default=0, type='int',
            help='do you wish to save the results? (1=True, 0=False) [default: %default]')
    parser.add_option('-v', '--verbose',dest='verbose', default=1, type='int',
            help='verbosity level (0 is minimum, 4 is maximum) [default: %default]')
    parser.add_option('--init_id', dest='init_id', default=1, type='int',
            help='init_id (changes random seed for multiple initializations) [default: %default]')
    return parser


def parser_add_mf_options(parser):
    group = optparse.OptionGroup(parser, "Mondrian forest options")
    group.add_option('--n_mondrians', dest='n_mondrians', default=10, type='int',
            help='number of trees in mondrian forest [default: %default]')
    group.add_option('--budget', dest='budget', default=-1, type='float',
            help='budget for mondrian tree prior [default: %default]' \
                    ' NOTE: budget=-1 will be treated as infinity')
    group.add_option('--discount_factor', dest='discount_factor', default=10, type='float',
            help='value of discount_factor parameter for HNSP (optype=class) [default: %default] '
            'NOTE: actual discount parameter = discount_factor * num_dimensions')
    group.add_option('--n_minibatches', dest='n_minibatches', default=1, type='int',
            help='number of minibatches [default: %default]')
    group.add_option('--draw_mondrian', dest='draw_mondrian', default=0, type='int',
            help='do you want to draw mondrians? (0=False, 1=True) [default: %default] ')
    group.add_option('--smooth_hierarchically', dest='smooth_hierarchically', default=1, type='int',
            help='do you want to smooth hierarchically? (0=False, 1=True)')
    group.add_option('--store_every', dest='store_every', default=0, type='int',
            help='do you want to store mondrians at every iteration? (0=False, 1=True)')
    group.add_option('--bagging', dest='bagging', default=0, type='int',
            help='do you want to use bagging? (0=False) [default: %default] ')
    group.add_option('--min_samples_split', dest='min_samples_split', default=2, type='int',
            help='the minimum number of samples required to split an internal node ' \
                    '(used only for optype=real) [default: %default]')
    parser.add_option_group(group)
    return parser


def parser_check_common_options(parser, settings):
    fail(parser, not(settings.save==0 or settings.save==1), 'save needs to be 0/1')
    fail(parser, not(settings.smooth_hierarchically==0 or settings.smooth_hierarchically==1), \
            'smooth_hierarchically needs to be 0/1')
    fail(parser, not(settings.normalize_features==0 or settings.normalize_features==1), 'normalize_features needs to be 0/1')
    fail(parser, not(settings.optype=='real' or settings.optype=='class'), 'optype needs to be real/class')


def parser_check_mf_options(parser, settings):
    fail(parser, settings.n_mondrians < 1, 'number of mondrians needs to be >= 1')
    fail(parser, settings.discount_factor <= 0, 'discount_factor needs to be > 0')
    fail(parser, not(settings.budget == -1 or settings.budget > 0), 'budget needs to be > 0 or -1 (treated as INF)')
    fail(parser, settings.n_minibatches < 1, 'number of minibatches needs to be >= 1')
    fail(parser, not(settings.draw_mondrian==0 or settings.draw_mondrian==1), 'draw_mondrian needs to be 0/1')
    fail(parser, not(settings.store_every==0 or settings.store_every==1), 'store_every needs to be 0/1')
    fail(parser, not(settings.bagging==0), 'bagging=1 not supported; please set bagging=0')
    fail(parser, settings.min_samples_split < 1, 'min_samples_split needs to be > 1')
    # added additional checks for MF
    if settings.normalize_features != 1:
        warn('normalize_features not equal to 1; mondrian forests assume that features are on the same scale')


def fail(parser, condition, msg):
    if condition:
        print msg
        print
        parser.print_help()
        sys.exit(1)


def reset_random_seed(settings):
    # Resetting random seed
    np.random.seed(settings.init_id * 1000)
    random.seed(settings.init_id * 1000)


def check_dataset(settings):
    classification_datasets = set(['satimage', 'usps', 'dna', 'dna-61-120', 'letter'])
    regression_datasets = set(['housing', 'kin40k'])
    special_cases = settings.dataset[:3] == 'toy' or settings.dataset[:4] == 'rsyn' \
            or settings.dataset[:8] == 'ctslices' or settings.dataset[:3] == 'msd' \
            or settings.dataset[:6] == 'houses' or settings.dataset[:9] == 'halfmoons' \
            or settings.dataset[:3] == 'sim' or settings.dataset == 'navada' \
            or settings.dataset[:3] == 'msg' or settings.dataset[:14] == 'airline-delays' \
            or settings.dataset == 'branin'
    if not special_cases:
        try:
            if settings.optype == 'class':
                assert(settings.dataset in classification_datasets)
            else:
                assert(settings.dataset in regression_datasets)
        except AssertionError:
            print 'Invalid dataset for optype; dataset = %s, optype = %s' % \
                    (settings.dataset, settings.optype)
            raise AssertionError
    return special_cases


def load_data(settings):
    data = {}
    special_cases = check_dataset(settings)
    if not special_cases:
        data = pickle.load(open(settings.data_path + settings.dataset + '/' + \
                settings.dataset + '.p', "rb"))
    elif settings.dataset == 'toy-mf':
        data = load_toy_mf_data()
    elif settings.dataset == 'msg-4dim':
        data = load_msg_data()
    elif settings.dataset[:9] == 'halfmoons':
        data = load_halfmoons(settings.dataset)
    elif settings.dataset[:4] == 'rsyn' or settings.dataset[:8] == 'ctslices' \
            or settings.dataset[:6] == 'houses' or settings.dataset[:3] == 'msd':
        data = load_rgf_datasets(settings)
    elif settings.dataset[:13] == 'toy-hypercube':
        n_dim = int(settings.dataset[14:])
        data = load_toy_hypercube(n_dim, settings, settings.optype == 'class')
    elif settings.dataset[:14] == 'airline-delays':
        filename = settings.data_path + 'airline-delays/' + settings.dataset + '.p'
        data = pickle.load(open(filename, 'rb'))
    else:
        print 'Unknown dataset: ' + settings.dataset
        raise Exception
    assert(not data['is_sparse'])
    try:
        if settings.normalize_features == 1:
            min_d = np.minimum(np.min(data['x_train'], 0), np.min(data['x_test'], 0))
            max_d = np.maximum(np.max(data['x_train'], 0), np.max(data['x_test'], 0))
            range_d = max_d - min_d
            idx_range_d_small = range_d <= 0.   # find columns where all features are identical
            if data['n_dim'] > 1:
                range_d[idx_range_d_small] = 1e-3   # non-zero value just to prevent division by 0
            elif idx_range_d_small:
                range_d = 1e-3
            data['x_train'] -= min_d + 0.
            data['x_train'] /= range_d
            data['x_test'] -= min_d + 0.
            data['x_test'] /= range_d
    except AttributeError:
        # backward compatibility with code without normalize_features argument
        pass
    if settings.select_features:
        if settings.optype == 'real':
            scores, _ = feature_selection.f_regression(data['x_train'], data['y_train'])
        scores[np.isnan(scores)] = 0.   # FIXME: setting nan scores to 0. Better alternative?
        scores_sorted, idx_sorted = np.sort(scores), np.argsort(scores)
        flag_relevant = scores_sorted > (scores_sorted[-1] * 0.05)  # FIXME: better way to set threshold?
        idx_feat_selected = idx_sorted[flag_relevant]
        assert len(idx_feat_selected) >= 1
        print scores
        print scores_sorted
        print idx_sorted
        # plt.plot(scores_sorted)
        # plt.show()
        if False:
            data['x_train'] = data['x_train'][:, idx_feat_selected]
            data['x_test'] = data['x_test'][:, idx_feat_selected]
        else:
            data['x_train'] = np.dot(data['x_train'], np.diag(scores))
            data['x_test'] = np.dot(data['x_test'], np.diag(scores))
        data['n_dim'] = data['x_train'].shape[1]
    # ------ beginning of hack ----------
    is_mondrianforest = True
    n_minibatches = settings.n_minibatches
    if is_mondrianforest:
        # creates data['train_ids_partition']['current'] and data['train_ids_partition']['cumulative']
        #    where current[idx] contains train_ids in minibatch "idx", cumulative contains train_ids in all
        #    minibatches from 0 till idx  ... can be used in gen_train_ids_mf or here (see below for idx > -1)
        data['train_ids_partition'] = {'current': {}, 'cumulative': {}}
        train_ids = np.arange(data['n_train'])
        try:
            draw_mondrian = settings.draw_mondrian
        except AttributeError:
            draw_mondrian = False
        if is_mondrianforest and (not draw_mondrian):
            reset_random_seed(settings)
            np.random.shuffle(train_ids)
            # NOTE: shuffle should be the first call after resetting random seed
            #       all experiments would NOT use the same dataset otherwise
        train_ids_cumulative = np.arange(0)
        n_points_per_minibatch = data['n_train'] / n_minibatches
        assert n_points_per_minibatch > 0
        idx_base = np.arange(n_points_per_minibatch)
        for idx_minibatch in range(n_minibatches):
            is_last_minibatch = (idx_minibatch == n_minibatches - 1)
            idx_tmp = idx_base + idx_minibatch * n_points_per_minibatch
            if is_last_minibatch:
                # including the last (data[n_train'] % settings.n_minibatches) indices along with indices in idx_tmp
                idx_tmp = np.arange(idx_minibatch * n_points_per_minibatch, data['n_train'])
            train_ids_current = train_ids[idx_tmp]
            # print idx_minibatch, train_ids_current
            data['train_ids_partition']['current'][idx_minibatch] = train_ids_current
            train_ids_cumulative = np.append(train_ids_cumulative, train_ids_current)
            data['train_ids_partition']['cumulative'][idx_minibatch] = train_ids_cumulative
    return data


def get_correlation(X, y):
    scores = np.zeros(X.shape[1])
    for i_col in np.arange(X.shape[1]):
        x = X[:, i_col]
        scores[i_col] = np.abs(pearsonr(x, y)[0])
    return scores


def load_toy_hypercube(n_dim, settings, class_output=False):
    n_train = n_test = 10 * (2 ** n_dim)
    reset_random_seed(settings)
    x_train, y_train, f_train, f_values = gen_hypercube_data(n_train, n_dim, class_output)
    x_test, y_test, f_test, f_values = gen_hypercube_data(n_test, n_dim, class_output, f_values)
    data = {'x_train': x_train, 'y_train': y_train, \
            'f_train': f_train, 'f_test': f_test, \
            'n_dim': n_dim, 'n_train': n_train, 'x_test': x_test, \
            'y_test': y_test, 'n_test': n_test, 'is_sparse': False}
    if class_output:
        data['n_class'] = 2 ** n_dim
    return data


def gen_hypercube_data(n_points, n_dim, class_output, f_values=None):
    # synthetic hypercube-like dataset
    # x-values of data points are close to vertices of a hypercube
    # y-value of data point is different
    y_sd = 0.
    x_sd = 0.1
    mag = 3
    x = x_sd * np.random.randn(n_points, n_dim)
    n_vertices = 2 ** n_dim
    #i = np.random.randint(0, n_vertices, n_points)
    i = np.arange(n_vertices).repeat(n_points / n_vertices)     # equal distribution
    offsets = np.zeros((n_vertices, n_dim))
    for d in range(n_dim):
        tmp = np.ones(2**(n_dim-d))
        tmp[:2**(n_dim-d-1)] = -1
        offsets[:, d] = np.tile(tmp, (1, 2**d))[0]
    x += offsets[i, :]
    y = np.zeros(n_points)
    #f = np.zeros(n_points)
    if class_output:
        y = f = i
    else:
        if f_values is None:
            # generate only for training data
            f_values = np.random.randn(n_vertices) * mag
        f = f_values[i]
        y = f + y_sd * np.random.randn(n_points)
    return (x, y, f, f_values)


def load_msg_data():
    mat = scipy.io.loadmat('wittawat/demo_uncertainty_msgs_4d.mat')
    x_test = np.vstack((mat['Xte1'], mat['Xte2']))
    n_test1 = mat['Xte1'].shape[0]
    y_test = np.nan * np.ones(x_test.shape[0])
    data = {'x_train': mat['Xtr'], 'y_train': np.ravel(mat['Ytr']), \
            'x_test': x_test, 'y_test': y_test, \
            'x_test1': mat['Xte1'], 'x_test2': mat['Xte2'], \
            'n_test1': n_test1, \
            'y_test1': y_test, 'y_test2': y_test, \
            'n_train': mat['Xtr'].shape[0], 'n_test': x_test.shape[0], \
            'n_dim': x_test.shape[1], 'is_sparse': False}
    return data


def add_stuff_2_settings(settings):
    settings.perf_dataset_keys = ['train', 'test']
    if settings.optype == 'class':
        settings.perf_store_keys = ['pred_prob']
        settings.perf_metrics_keys = ['log_prob', 'acc']
    else:
        settings.perf_store_keys = ['pred_mean', 'pred_prob']
        settings.perf_metrics_keys = ['log_prob', 'mse']
    settings.name_metric = get_name_metric(settings)


def get_name_metric(settings):
    name_metric = settings.perf_metrics_keys[1]
    assert(name_metric == 'mse' or name_metric == 'acc')
    return name_metric


def load_toy_data():
    n_dim = 2
    n_train_pc = 4
    n_class = 2
    n_train = n_train_pc * n_class
    n_test = n_train
    y_train = np.r_[np.ones(n_train_pc, dtype='int'), \
            np.zeros(n_train_pc, dtype='int')]
    y_test = np.r_[np.ones(n_train_pc, dtype='int'), \
            np.zeros(n_train_pc, dtype='int')]
    x_train = np.random.randn(n_train, n_dim)
    x_test = np.random.randn(n_train, n_dim)
    mag = 5
    for i, y_ in enumerate(y_train):
        if y_ == 0:
            x_train[i, :] += np.sign(np.random.rand() - 0.5) * mag
        else:
            tmp = np.sign(np.random.rand() - 0.5)
            x_train[i, :] += np.array([tmp, -tmp]) * mag
    for i, y_ in enumerate(y_test):
        if y_ == 0:
            x_test[i, :] += np.sign(np.random.rand() - 0.5) * mag
        else:
            tmp = np.sign(np.random.rand() - 0.5)
            x_test[i, :] += np.array([tmp, -tmp]) * mag
    data = {'x_train': x_train, 'y_train': y_train, 'n_class': n_class, \
            'n_dim': n_dim, 'n_train': n_train, 'x_test': x_test, \
            'y_test': y_test, 'n_test': n_test, 'is_sparse': False}
    print data
    return data


def load_toy_mf_data():
    n_dim = 2
    n_class = 3
    x_train = np.array([-0.5,-1, -2,-2, 1,0.5, 2,2, -1,1, -1.5, 1.5]) + 0.
    y_train = np.array([0, 0, 1, 1, 2, 2], dtype='int')
    x_train.shape = (6, 2)
    if False:
        plt.figure()
        plt.hold(True)
        plt.scatter(x_train[:2, 0], x_train[:2, 1], color='b')
        plt.scatter(x_train[2:4, 0], x_train[2:4, 1], color='r')
        plt.scatter(x_train[4:, 0], x_train[4:, 1], color='k')
        plt.savefig('toy-mf_dataset.pdf', type='pdf')
    x_test = x_train
    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    y_test = np.array([0, 0, 1, 1, 2, 2], dtype='int')
    data = {'x_train': x_train, 'y_train': y_train, 'n_class': n_class, \
            'n_dim': n_dim, 'n_train': n_train, 'x_test': x_test, \
            'y_test': y_test, 'n_test': n_test, 'is_sparse': False}
    return data


def load_halfmoons(dataset):
    n_dim = 2
    n_class = 2
    if dataset == 'halfmoons':
        x_train = np.array([-3,0, -2,1, -1,2, 0,3, 1,2, 2,1, 3,0, -1.5,1.5, -0.5,0.5, 0.5,-0.5, 1.5,-1.5, 2.5,-0.5, 3.5,0.5, 4.5,1.5]) + 0.
        y_train = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype='int')
        x_train.shape = (14, 2)
        x_train[7:, 0] += 1.5
        x_train[7:, 1] -= 0.5
        i1 = np.arange(7)
        i2 = np.arange(7, 14)
    elif dataset == 'halfmoons2':
        n = 500
        x_train = np.random.rand(n, 2)
        i1 = np.arange(n / 2)
        i2 = np.arange(n / 2, n)
        x_train[i1, 0] = 2 * x_train[i1, 0] - 1
        x_train[i2, 0] = 2 * x_train[i2, 0]
        x_train[i1, 1] = 1 - x_train[i1, 0] * x_train[i1, 0]
        x_train[i2, 1] = (x_train[i2, 0] - 1) * (x_train[i2, 0] - 1) - 0.5
        x_train[:, 1] += 0.1 * np.random.randn(n)
        y_train = np.zeros(n, dtype='int')
        y_train[i2] = 1
    else:
        raise Exception
    if False:
        plt.figure()
        plt.hold(True)
        plt.scatter(x_train[i1, 0], x_train[i1, 1], color='b')
        plt.scatter(x_train[i2, 0], x_train[i2, 1], color='r')
        name = '%s_dataset.pdf' % dataset
        plt.savefig(name, type='pdf')
    x_test, y_test = x_train.copy(), y_train.copy()
    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    data = {'x_train': x_train, 'y_train': y_train, 'n_class': n_class, \
            'n_dim': n_dim, 'n_train': n_train, 'x_test': x_test, \
            'y_test': y_test, 'n_test': n_test, 'is_sparse': False}
    return data


def load_rgf_datasets(settings):
    filename_train = settings.data_path + 'exp-data' + '/' + settings.dataset
    filename_test = filename_train[:-3]
    x_train = np.loadtxt(filename_train + '.train.x')
    y_train = np.loadtxt(filename_train + '.train.y')
    x_test = np.loadtxt(filename_test + '.test.x')
    y_test = np.loadtxt(filename_test + '.test.y')
    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    n_dim = x_train.shape[1]
    data = {'x_train': x_train, 'y_train': y_train, \
            'n_dim': n_dim, 'n_train': n_train, 'x_test': x_test, \
            'y_test': y_test, 'n_test': n_test, 'is_sparse': False}
    return data


def get_tree_limits(p, data):
    d_extent = {}
    x = data['x_train']
    x1_min = np.min(x[:, 1])
    x1_max = np.max(x[:, 1])
    x0_min = np.min(x[:, 0])
    x0_max = np.max(x[:, 0])
    d_extent[0] = (x0_min, x0_max, x1_min, x1_max)
    if not p.node_info:
        return ([], [])
    non_leaf_max = max(p.node_info.keys())
    p_d = p.node_info
    hlines_list = []
    vlines_list = []
    for node_id in range(non_leaf_max + 1):
        if node_id not in p.node_info:
            continue
        x0_min, x0_max, x1_min, x1_max = d_extent[node_id]
        print p_d[node_id]
        feat_id, split, idx_split_global = p_d[node_id]
        if feat_id == 0:
            vlines_list.append([split, x1_min, x1_max])
            left_extent = (x0_min, split, x1_min, x1_max)
            right_extent = (split, x0_max, x1_min, x1_max)
        else:
            hlines_list.append([split, x0_min, x0_max])
            left_extent = (x0_min, x0_max, x1_min, split)
            right_extent = (x0_min, x0_max, split, x1_max)
        left, right = get_children_id(node_id)
        d_extent[left] = left_extent
        d_extent[right] = right_extent
    return (hlines_list, vlines_list)


def bootstrap(train_ids, settings=None):
    """ online bagging: each point is included Poisson(1) times """
    n = len(train_ids)
    cnt_all = np.random.poisson(1, n)
    op = []
    for train_id, cnt in izip(train_ids, cnt_all):
        op.extend([train_id] * cnt)
    return np.array(op)


class Param(object):
    def __init__(self, settings):
        self.budget = settings.budget


def get_filename_mf(settings):
    if settings.optype == 'class':
        param_str = '%s' % settings.alpha
    else:
        param_str = ''
    split_str = 'mf-budg-%s_nmon-%s_mini-%s_discount-%s' % (settings.budget, settings.n_mondrians, \
                                        settings.n_minibatches, settings.discount_factor)
    filename = settings.op_dir + '/' + '%s-%s-param-%s-init_id-%s-bag-%s-tag-%s.p' % \
            (settings.dataset, split_str, param_str, settings.init_id, \
                settings.bagging, settings.tag)
    return filename


def create_prediction_tree(tree, param, data, settings, all_nodes=False):
    init_prediction_tree(tree, settings)
    for node_id in tree.leaf_nodes:
        update_predictive_posterior_node(tree, param, data, settings, node_id)
    if all_nodes:
        for node_id in tree.non_leaf_nodes:
            update_predictive_posterior_node(tree, param, data, settings, node_id)


def init_prediction_tree(tree, settings):
    if settings.optype == 'class':
        tree.pred_prob = {}


def update_predictive_posterior_node(tree, param, data, settings, node_id):
    if settings.optype == 'class':
        tmp = tree.counts[node_id] + param.alpha_vec
        tree.pred_prob[node_id] = tmp / float(tmp.sum())
    else:
        tree.pred_mean[node_id] = tree.sum_y[node_id] / float(tree.n_points[node_id])


def compute_metrics_classification(y_test, pred_prob, do_not_compute_log_prob=False):
    acc, log_prob = 0.0, 0.0
    for n, y in enumerate(y_test):
        tmp = pred_prob[n, :]
        #pred = np.argmax(tmp)
        pred = random.choice(np.argwhere(tmp == np.amax(tmp)).flatten())    # randomly break ties
        acc += (pred == y)
        if not do_not_compute_log_prob:
            log_tmp_pred = math.log(tmp[y])
            try:
                assert(not np.isinf(abs(log_tmp_pred)))
            except AssertionError:
                'print abs(log_tmp_pred) = inf in compute_metrics_classification; tmp = '
                print tmp
                raise AssertionError
            log_prob += log_tmp_pred
    acc /= (n + 1)
    if not do_not_compute_log_prob:
        log_prob /= (n + 1)
    else:
        log_prob = -np.inf
    metrics = {'acc': acc, 'log_prob': log_prob}
    return metrics


def test_compute_metrics_classification():
    n = 100
    n_class = 10
    pred_prob = np.random.rand(n, n_class)
    y = np.ones(n)
    metrics = compute_metrics_classification(y, pred_prob)
    print 'chk if same: %s, %s' % (metrics['log_prob'], np.mean(np.log(pred_prob[:, 1])))
    assert(np.abs(metrics['log_prob']  - np.mean(np.log(pred_prob[:, 1]))) < 1e-10)
    pred_prob[:, 1] = 1e5
    metrics = compute_metrics_classification(y, pred_prob)
    assert np.abs(metrics['acc'] - 1) < 1e-3
    print 'chk if same: %s, 1.0' % (metrics['acc'])


def compute_metrics_regression(y_test, pred_mean, pred_prob=None):
    # print 'y_test: ', y_test[:5]
    # print 'pred_mean: ', pred_mean[:5]
    mse = np.mean((y_test - pred_mean) ** 2)
    log_prob = np.mean(np.log(pred_prob))
    metrics = {'mse': mse, 'log_prob': log_prob}
    return metrics


def test_compute_metrics_regression():
    n = 100
    pred_prob = np.random.rand(n)
    y = np.random.randn(n)
    pred = np.ones(n)
    metrics = compute_metrics_regression(y, pred, pred_prob)
    print 'chk if same: %s, %s' % (metrics['mse'], np.mean((y - 1) ** 2))
    assert np.abs(metrics['mse'] - np.mean((y - 1) ** 2)) < 1e-3


def is_split_valid(split_chosen, x_min, x_max):
    try:
        assert(split_chosen > x_min)
        assert(split_chosen < x_max)
    except AssertionError:
        print 'split_chosen <= x_min or >= x_max'
        raise AssertionError


def evaluate_performance_tree(p, param, data, settings, x_test, y_test):
    create_prediction_tree(p, param, data, settings)
    pred_all = evaluate_predictions_fast(p, x_test, y_test, data, param, settings)
    pred_prob = pred_all['pred_prob']
    if settings.optype == 'class':
        metrics = compute_metrics_classification(y_test, pred_prob)
    else:
        pred_mean = pred_all['pred_mean']
        metrics = compute_metrics_regression(y_test, pred_mean, pred_prob)
    return (metrics)


def stop_split(train_ids, settings, data, cache):
    if (len(train_ids) <= settings.min_size):
        op = True
    else:
        op = no_valid_split_exists(data, cache, train_ids, settings)
    return op


def compute_dirichlet_normalizer(cnt, alpha=0.0, prior_term=None):
    """ cnt is np.array, alpha is concentration of Dirichlet prior
        => alpha/K is the mass for each component of a K-dimensional Dirichlet
    """
    try:
        assert(len(cnt.shape) == 1)
    except AssertionError:
        print 'cnt should be a 1-dimensional np array'
        raise AssertionError
    n_class = float(len(cnt))
    if prior_term is None:
        #print 'recomputing prior_term'
        prior_term = gammaln(alpha) - n_class * gammaln(alpha / n_class)
    op = np.sum(gammaln(cnt + alpha / n_class)) - gammaln(np.sum(cnt) + alpha) \
            + prior_term
    return op


def compute_dirichlet_normalizer_fast(cnt, cache):
    """ cnt is np.array, alpha is concentration of Dirichlet prior
        => alpha/K is the mass for each component of a K-dimensional Dirichlet
    """
    op = compute_gammaln_1(cnt, cache) - compute_gammaln_2(cnt.sum(), cache) \
            + cache['alpha_prior_term']
    return op


def evaluate_predictions(p, x, y, data, param):
    (pred, pred_prob) = p.predict(x, data['n_class'], param.alpha)
    (acc, log_prob) = compute_metrics(y, pred_prob)
    return (pred, pred_prob, acc, log_prob)


def init_left_right_statistics():
    return(None, None, {}, -np.inf, -np.inf)


def compute_left_right_statistics(data, param, cache, train_ids, feat_id_chosen, \
        split_chosen, settings):
    cond = data['x_train'][train_ids, feat_id_chosen] <= split_chosen
    train_ids_left = train_ids[cond]
    train_ids_right = train_ids[~cond]
    cache_tmp = {}
    if settings.optype == 'class':
        range_n_class = cache['range_n_class']
	#ipdb.set_trace()
        cnt_left_chosen = np.bincount(data['y_train'][train_ids_left], minlength=data['n_class'])
        cnt_right_chosen = np.bincount(data['y_train'][train_ids_right], minlength=data['n_class'])
        cache_tmp['cnt_left_chosen'] = cnt_left_chosen
        cache_tmp['cnt_right_chosen'] = cnt_right_chosen
    else:
        cache_tmp['sum_y_left'] = np.sum(data['y_train'][train_ids_left])
        cache_tmp['sum_y2_left'] = np.sum(data['y_train'][train_ids_left] ** 2)
        cache_tmp['n_points_left'] = len(train_ids_left)
        cache_tmp['sum_y_right'] = np.sum(data['y_train'][train_ids_right])
        cache_tmp['sum_y2_right'] = np.sum(data['y_train'][train_ids_right] ** 2)
        cache_tmp['n_points_right'] = len(train_ids_right)
    if settings.verbose >= 2:
        print 'feat_id_chosen = %s, split_chosen = %s' % (feat_id_chosen, split_chosen)
        print 'y (left) = %s\ny (right) = %s' % (data['y_train'][train_ids_left], \
                                                    data['y_train'][train_ids_right])
    return(train_ids_left, train_ids_right, cache_tmp)


def get_reg_stats(y):
    # y is a list of numbers, get_reg_stats(y) returns stats required for computing regression likelihood
    y_ = np.array(y)
    sum_y = float(np.sum(y_))
    n_points = len(y_)
    sum_y2 = float(np.sum(pow(y_, 2)))
    return (sum_y, sum_y2, n_points)


def compute_entropy(cnts, alpha=0.0):
    """ returns the entropy of a multinomial distribution with
        mean parameter \propto (cnts + alpha/len(cnts))
        entropy unit = nats """
    prob = cnts * 1.0 + alpha / len(cnts)
    prob /= float(np.sum(prob))
    entropy = 0.0
    for k in range(len(cnts)):
        if abs(prob[k]) > 1e-12:
            entropy -= prob[k] * np.log(prob[k])
    return entropy


def precompute_minimal(data, settings):
    param = empty()
    cache = {}
    if settings.optype == 'class':
        param.alpha = settings.alpha
        param.alpha_per_class = float(param.alpha) / data['n_class']
        cache['y_train_counts'] = hist_count(data['y_train'], range(data['n_class']))
        cache['range_n_class'] = range(data['n_class'])
        param.base_measure = (np.ones(data['n_class']) + 0.) / data['n_class']
        param.alpha_vec = param.base_measure * param.alpha
    else:
        cache['sum_y'] = float(np.sum(data['y_train']))
        cache['sum_y2'] = float(np.sum(data['y_train'] ** 2))
        cache['n_points'] = len(data['y_train'])
        warn('initializing prior mean and precision to their true values')
        # FIXME: many of the following are relevant only for mondrian forests
        param.prior_mean = np.mean(data['y_train'])
        param.prior_variance = np.var(data['y_train'])
        param.prior_precision = 1.0 / param.prior_variance
        if not settings.smooth_hierarchically:
            param.noise_variance = 0.01     # FIXME: hacky
        else:
            K = min(1000, data['n_train'])     # FIXME: measurement noise set to fraction of unconditional variance
            param.noise_variance = param.prior_variance / (1. + K)  # assume noise variance = prior_variance / (2K)
            # NOTE: max_split_cost scales inversely with the number of dimensions
        param.variance_coef = 2.0 * param.prior_variance
        param.sigmoid_coef = data['n_dim']  / (2.0 * np.log2(data['n_train']))
        param.noise_precision = 1.0 / param.noise_variance
    return (param, cache)


def init_update_posterior_node_incremental(tree, data, param, settings, cache, node_id, train_ids_new, \
         init_node_id=None):
    # init with sufficient statistics of init_node_id and add contributions of train_ids_new
    if settings.optype == 'class':
        if init_node_id is None:
            tree.counts[node_id] = 0
        else:
            tree.counts[node_id] = tree.counts[init_node_id] + 0
    else:
        if init_node_id is None:
            tree.sum_y[node_id] = 0
            tree.sum_y2[node_id] = 0
            tree.n_points[node_id] = 0
        else:
            tree.sum_y[node_id] = tree.sum_y[init_node_id] + 0
            tree.sum_y2[node_id] = tree.sum_y2[init_node_id] + 0
            tree.n_points[node_id] = tree.n_points[init_node_id] + 0

    update_posterior_node_incremental(tree, data, param, settings, cache, node_id, train_ids_new)


def update_posterior_node_incremental(tree, data, param, settings, cache, node_id, train_ids_new):
    y_train_new = data['y_train'][train_ids_new]
    if settings.optype == 'class':
	#ipdb.set_trace()
        tree.counts[node_id] += np.bincount(y_train_new, minlength=data['n_class'])
    else:
        sum_y_new, sum_y2_new, n_points_new = get_reg_stats(y_train_new)
        tree.sum_y[node_id] += sum_y_new
        tree.sum_y2[node_id] += sum_y2_new
        tree.n_points[node_id] += n_points_new
########################


def process_command_line(cmd_options):
    parser = parser_add_common_options()
    parser = parser_add_mf_options(parser)
    settings, args = parser.parse_args(cmd_options)
    add_stuff_2_settings(settings)
    if settings.optype == 'class':
        settings.alpha = 0    # normalized stable prior
        assert settings.smooth_hierarchically
    parser_check_common_options(parser, settings)
    parser_check_mf_options(parser, settings)
    if settings.budget < 0:
        settings.budget_to_use = INF
    else:
        settings.budget_to_use = settings.budget
    #reset_random_seed(settings)
    return settings


class MondrianBlock(object):
    """
    defines Mondrian block
    variables:
    - min_d         : dimension-wise min of training data in current block
    - max_d         : dimension-wise max of training data in current block
    - range_d       : max_d - min_d
    - sum_range_d   : sum of range_d
    - left          : id of left child
    - right         : id of right child
    - parent        : id of parent
    - is_leaf       : boolen variable to indicate if current block is leaf
    - budget        : remaining lifetime for subtree rooted at current block
                      = lifetime of Mondrian - time of split of parent
                      NOTE: time of split of parent of root node is 0
    """
    def __init__(self, data, settings, budget, parent, range_stats):
        self.min_d, self.max_d, self.range_d, self.sum_range_d = range_stats
        self.budget = budget + 0.
        self.parent = parent
        self.left = None
        self.right = None
        self.is_leaf = True


class MondrianTree(object):
    """
    defines a Mondrian tree
    variables:
    - node_info     : stores splits for internal nodes
    - root          : id of root node
    - leaf_nodes    : list of leaf nodes
    - non_leaf_nodes: list of non-leaf nodes
    - max_split_costs   : max_split_cost for a node is time of split of node - time of split of parent
                          max_split_cost is drawn from an exponential
    - train_ids     : list of train ids stored for paused Mondrian blocks
    - counts        : stores histogram of labels at each node (when optype = 'class')
    - grow_nodes    : list of Mondrian blocks that need to be "grown"
    functions:
    - __init__      : initialize a Mondrian tree
    - grow          : samples Mondrian block (more precisely, restriction of blocks to training data)
    - extend_mondrian   : extend a Mondrian to include new training data
    - extend_mondrian_block : conditional Mondrian algorithm
    """
    def __init__(self, data=None, train_ids=None, settings=None, param=None, cache=None):
        """
        initialize Mondrian tree data structure and sample restriction of Mondrian tree to current training data
        data is a N x D numpy array containing the entire training data
        train_ids is the training ids of the first minibatch
        """
        if data is None:
            return
        root_node = MondrianBlock(data, settings, settings.budget_to_use, None, \
                        get_data_range(data, train_ids))
        self.root = root_node
        self.non_leaf_nodes = []
        self.leaf_nodes = []
        self.node_info = {}
        self.max_split_costs = {}
        self.split_times = {}
        self.train_ids = {root_node: train_ids}
        self.copy_params(param, settings)
        init_prediction_tree(self, settings)
        if cache:
            if settings.optype == 'class':
                #ipdb.set_trace()
                self.counts = {root_node: cache['y_train_counts']}
            else:
                self.sum_y = {root_node: cache['sum_y']}
                self.sum_y2 = {root_node: cache['sum_y2']}
                self.n_points = {root_node: cache['n_points']}
            if settings.bagging == 1 or settings.n_minibatches > 1:
                init_update_posterior_node_incremental(self, data, param, settings, cache, root_node, train_ids)
        self.grow_nodes = [root_node]
        self.grow(data, settings, param, cache)

    def copy_params(self, param, settings):
        if settings.optype == 'real':
            self.noise_variance = param.noise_variance + 0
            self.noise_precision = param.noise_precision + 0
            self.sigmoid_coef = param.sigmoid_coef + 0
            self.variance_coef = param.variance_coef + 0

    def get_average_depth(self, settings, data):
        """
        compute average depth of tree (averaged over training data)
        = depth of a leaf weighted by fraction of training data at that leaf
        """
        self.depth_nodes = {self.root: 0}
        tmp_node_list = [self.root]
        n_total = 0.
        average_depth = 0.
        self.node_size_by_depth = defaultdict(list)
        leaf_node_sizes = []
        while True:
            try:
                node_id = tmp_node_list.pop(0)
            except IndexError:
                break
            if node_id.is_leaf:
                if settings.optype == 'class':
                    n_points_node = np.sum(self.counts[node_id])
                else:
                    n_points_node = self.n_points[node_id]
                n_total += n_points_node
                average_depth += n_points_node * self.depth_nodes[node_id]
                self.node_size_by_depth[self.depth_nodes[node_id]].append(node_id.sum_range_d)
            if not node_id.is_leaf:
                self.depth_nodes[node_id.left] = self.depth_nodes[node_id] + 1
                self.depth_nodes[node_id.right] = self.depth_nodes[node_id] + 1
                tmp_node_list.extend([node_id.left, node_id.right])
            else:
                leaf_node_sizes.append(node_id.sum_range_d)
        #assert data['n_train'] == int(n_total)
        average_depth /= n_total
        average_leaf_node_size = np.mean(leaf_node_sizes)
        average_node_size_by_depth = {}
        for k in self.node_size_by_depth:
            average_node_size_by_depth[k] = np.mean(self.node_size_by_depth[k])
        return (average_depth, average_leaf_node_size, average_node_size_by_depth)

    def get_print_label_draw_tree(self, node_id, graph):
        """
        helper function for draw_tree using pydot
        """
        name = self.node_ids_print[node_id]
        name2 = name
        if name2 == '':
            name2 = 'e'
        if node_id.is_leaf:
            op = name
        else:
            feat_id, split = self.node_info[node_id]
            op = r'x_%d > %.2f\nt = %.2f' % (feat_id+1, split, self.cumulative_split_costs[node_id])
        if op == '':
            op = 'e'
        node = pydot.Node(name=name2, label=op) # latex labels don't work
        graph.add_node(node)
        return (name2, graph)

    def draw_tree(self, data, settings, figure_id=0, i_t=0):
        """
        function to draw Mondrian tree using pydot
        NOTE: set ADD_TIME=True if you want want set edge length between parent and child
                to the difference in time of splits
        """
        self.gen_node_ids_print()
        self.gen_cumulative_split_costs_only(settings, data)
        graph = pydot.Dot(graph_type='digraph')
        dummy, graph = self.get_print_label_draw_tree(self.root, graph)
        ADD_TIME = False
        for node_id in self.non_leaf_nodes:
            parent, graph = self.get_print_label_draw_tree(node_id, graph)
            left, graph = self.get_print_label_draw_tree(node_id.left, graph)
            right, graph = self.get_print_label_draw_tree(node_id.right, graph)
            for child, child_id in izip([left, right], [node_id.left, node_id.right]):
                edge = pydot.Edge(parent, child)
                if ADD_TIME and (not child_id.is_leaf):
                    edge.set_minlen(self.max_split_costs[child_id])
                    edge2 = pydot.Edge(dummy, child)
                    edge2.set_minlen(self.cumulative_split_costs[child_id])
                    edge2.set_style('invis')
                    graph.add_edge(edge2)
                graph.add_edge(edge)
        filename_plot_tag = get_filename_mf(settings)[:-2]
        if settings.save:
            tree_name = filename_plot_tag + '-mtree_minibatch-' + str(figure_id) + '.pdf'
            print 'saving file: %s' % tree_name
            graph.write_pdf(tree_name)

    def draw_mondrian(self, data, settings, figure_id=None, i_t=0):
        """
        function to draw Mondrian partitions; each Mondrian tree is one subplot.
        """
        assert data['n_dim'] == 2 and settings.normalize_features == 1 \
                and settings.n_mondrians <= 10
        self.gen_node_list()
        if settings.n_mondrians == 1 and settings.dataset == 'toy-mf':
            self.draw_tree(data, settings, figure_id, i_t)
        if settings.n_mondrians > 2:
            n_row = 2
        else:
            n_row = 1
        n_col = int(math.ceil(settings.n_mondrians / n_row))
        if figure_id is None:
            figure_id = 0
        fig = plt.figure(figure_id)
        plt.hold(True)
        ax = plt.subplot(n_row, n_col, i_t+1, aspect='equal')
        EPS = 0.
        ax.set_xlim(xmin=0-EPS)
        ax.set_xlim(xmax=1+EPS)
        ax.set_ylim(ymin=0-EPS)
        ax.set_ylim(ymax=1+EPS)
        ax.autoscale(False)
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        non_leaf_nodes = [self.root]
        while non_leaf_nodes:
            node_id = non_leaf_nodes.pop(0)
            try:
                feat_id, split = self.node_info[node_id]
            except:
                continue
            left, right = node_id.left, node_id.right
            non_leaf_nodes.append(left)
            non_leaf_nodes.append(right)
            EXTRA = 0.0    # to show splits that separate 2 data points
            if feat_id == 1:
                # axhline doesn't work if you rescale
                ax.hlines(split, node_id.min_d[0] - EXTRA, node_id.max_d[0] + EXTRA, lw=LW, color='k')
            else:
                ax.vlines(split, node_id.min_d[1] - EXTRA, node_id.max_d[1] + EXTRA, lw=LW, color='k')
        # add "outer patch" that defines the extent (not data dependent)
        block = patches.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='gray', ls='dashed')
        ax.add_patch(block)
        for i_, node_id in enumerate(self.node_list):
            # plot only the block where Mondrian has been induced (limited by extent of training data)
            block = patches.Rectangle((node_id.min_d[0], node_id.min_d[1]), node_id.range_d[0], \
                    node_id.range_d[1], facecolor='white', edgecolor='gray')
            ax.add_patch(block)
        for i_, node_id in enumerate(self.leaf_nodes):
            # plot only the block where Mondrian has been induced (limited by extent of training data)
            block = patches.Rectangle((node_id.min_d[0], node_id.min_d[1]), node_id.range_d[0], \
                    node_id.range_d[1], facecolor=colors_list[i_ % len(colors_list)], edgecolor='black')
            ax.add_patch(block)
            # zorder = 1 will make points inside the blocks invisible, >= 2 will make them visible
            x_train = data['x_train'][self.train_ids[node_id], :]
            #ax.scatter(x_train[:, 0], x_train[:, 1], color='k', marker='x', s=10, zorder=2)
            color_y = 'rbk'
            for y_ in range(data['n_class']):
                idx = data['y_train'][self.train_ids[node_id]] == y_
                ax.scatter(x_train[idx, 0], x_train[idx, 1], color=color_y[y_], marker='o', s=16, zorder=2)
        plt.draw()

    def gen_node_ids_print(self):
        """
        generate binary string label for each node
        root_node is denoted by empty string "e"
        all other node labels are defined as follows: left(j) = j0, right(j) = j1
        e.g. left and right child of root_node are 0 and 1 respectively,
             left and right of node 0 are 00 and 01 respectively and so on.
        """
        node_ids = [self.root]
        self.node_ids_print = {self.root: ''}
        while node_ids:
            node_id = node_ids.pop(0)
            try:
                feat_id, split = self.node_info[node_id]
                left, right = node_id.left, node_id.right
                node_ids.append(left)
                node_ids.append(right)
                self.node_ids_print[left] = self.node_ids_print[node_id] + '0'
                self.node_ids_print[right] = self.node_ids_print[node_id] + '1'
            except KeyError:
                continue

    def print_dict(self, d):
        """
        print a dictionary
        """
        for k in d:
            print '\tk_map = %10s, val = %s' % (self.node_ids_print[k], d[k])

    def print_list(self, list_):
        """
        print a list
        """
        print '\t%s' % ([self.node_ids_print[x] for x in list_])

    def print_tree(self, settings):
        """
        prints some tree statistics: leaf nodes, non-leaf nodes, information and so on
        """
        self.gen_node_ids_print()
        print 'printing tree:'
        print 'len(leaf_nodes) = %s, len(non_leaf_nodes) = %s' \
                % (len(self.leaf_nodes), len(self.non_leaf_nodes))
        print 'node_info ='
        node_ids = [self.root]
        while node_ids:
            node_id = node_ids.pop(0)
            node_id_print = self.node_ids_print[node_id]
            try:
                feat_id, split = self.node_info[node_id]
                print '%10s, feat = %5d, split = %.2f, node_id = %s' % \
                        (node_id_print, feat_id, split, node_id)
                if settings.optype == 'class':
                    print 'counts = %s' % self.counts[node_id]
                else:
                    print 'n_points = %6d, sum_y = %.2f' % (self.n_points[node_id], self.sum_y[node_id])
                left, right = node_id.left, node_id.right
                node_ids.append(left)
                node_ids.append(right)
            except KeyError:
                continue
        print 'leaf info ='
        for node_id in self.leaf_nodes:
            node_id_print = self.node_ids_print[node_id]
            print '%10s, train_ids = %s, node_id = %s' % \
                    (node_id_print, self.train_ids[node_id], node_id)
            if settings.optype == 'class':
                print 'counts = %s' % self.counts[node_id]
            else:
                print 'n_points = %6d, sum_y = %.2f' % (self.n_points[node_id], self.sum_y[node_id])

    def check_if_labels_same(self, node_id):
        """
        checks if all labels in a node are identical
        """
        return np.count_nonzero(self.counts[node_id]) == 1

    def pause_mondrian(self, node_id, settings):
        """
        should you pause a Mondrian block or not?
        pause if sum_range_d == 0 (important for handling duplicates) or
        - optype == class: pause if all labels in a node are identical
        - optype == real: pause if n_points < min_samples_split
        """
        if settings.optype == 'class':
            pause_mondrian_tmp = self.check_if_labels_same(node_id)
        else:
            pause_mondrian_tmp = self.n_points[node_id] < settings.min_samples_split
        pause_mondrian = pause_mondrian_tmp or (node_id.sum_range_d == 0)
        return pause_mondrian

    def get_parent_split_time(self, node_id, settings):
        if node_id == self.root:
            return 0.
        else:
            return self.split_times[node_id.parent]

    def update_gaussian_hyperparameters(self, param, data, settings):
        n_points = float(self.n_points[self.root])
        param.prior_mean = self.sum_y[self.root] / n_points
        param.prior_variance = self.sum_y2[self.root] / n_points \
                                - param.prior_mean ** 2
        param.prior_precision = 1.0 / param.prior_variance
        # TODO: estimate K using estimate of noise variance at leaf nodes?
        # TODO: need to do this once for forest, rather than for each tree
        # FIXME very very hacky, surely a better way to tune this?
        if 'sfactor' in settings.tag:
            s_begin = settings.tag.find('sfactor-') + 8
            s_tmp = settings.tag[s_begin:]
            s_factor = float(s_tmp[:s_tmp.find('-')])
        else:
            s_factor = 2.0
        if 'kfactor' in settings.tag:
            k_begin = settings.tag.find('kfactor-') + 8
            k_tmp = settings.tag[k_begin:]
            k_factor = float(k_tmp[:k_tmp.find('-')])
        else:
            k_factor = min(2 * n_points, 500)  # noise variance is 1/K times prior_variance
        if k_factor <= 0.:
            K = 2. * n_points
        else:
            K = k_factor
        param.noise_variance = param.prior_variance / K
        param.noise_precision = 1.0 / param.noise_variance
        param.variance_coef = 2.0 * param.prior_variance * K / (K + 2.)
        param.sigmoid_coef = data['n_dim']  / (s_factor * np.log2(n_points))
        # FIXME: important to copy over since prediction accesses hyperparameters in self
        self.copy_params(param, settings)

    def get_node_mean_and_variance(self, node):
        n_points = float(self.n_points[node])
        node_mean = self.sum_y[node] / n_points
        node_variance = self.sum_y2[node] / n_points - node_mean ** 2
        return (node_mean, node_variance)

    def update_gaussian_hyperparameters_indep(self, param, data, settings):
        n_points = float(self.n_points[self.root])
        self.prior_mean, self.prior_variance = self.get_node_mean_and_variance(self.root)
        self.prior_precision = 1.0 / self.prior_variance
        self.cumulative_split_costs = {}
        self.leaf_means = []
        self.leaf_variances = []
        node_means = []
        d_node_means = {self.root: self.prior_mean}
        node_parent_means = []
        node_split_times = []
        node_parent_split_times = []
        if self.root.is_leaf:
            self.cumulative_split_costs[self.root] = 0.
            remaining = []
            self.max_split_time = 0.1   # NOTE: initial value, need to specify non-zero value
        else:
            self.cumulative_split_costs[self.root] = self.max_split_costs[self.root]
            remaining = [self.root.left, self.root.right]
            self.max_split_time = self.cumulative_split_costs[self.root] + 0
            node_split_times.append(self.cumulative_split_costs[self.root])
            node_parent_split_times.append(0.)
            node_means.append(self.prior_mean)
            node_parent_means.append(self.prior_mean)
        while True:
            try:
                node_id = remaining.pop(0)
            except IndexError:
                break
            self.cumulative_split_costs[node_id] = self.cumulative_split_costs[node_id.parent] \
                                                    + self.max_split_costs[node_id]
            node_mean, node_variance = self.get_node_mean_and_variance(node_id)
            node_split_times.append(self.cumulative_split_costs[node_id])
            node_parent_split_times.append(self.cumulative_split_costs[node_id.parent])
            node_means.append(node_mean)
            node_parent_means.append(d_node_means[node_id.parent])
            d_node_means[node_id] = node_mean
            if not node_id.is_leaf:
                remaining.append(node_id.left)
                remaining.append(node_id.right)
                self.max_split_time = max(self.max_split_time, self.cumulative_split_costs[node_id])
            else:
                self.leaf_means.append(node_mean)
                self.leaf_variances.append(node_variance)
        #self.noise_variance = np.max(self.leaf_variances)
        self.noise_variance = np.mean(self.leaf_variances)
        self.noise_precision = 1.0 / self.noise_variance
        self.sigmoid_coef = 3. / self.max_split_time
        #self.sigmoid_coef = data['n_dim']
        #self.sigmoid_coef = data['n_dim'] / 5
        #self.sigmoid_coef = data['n_dim']  / (2. * np.log2(n_points))
        #self.sigmoid_coef = data['n_dim']  / (2. * np.log2(n_points))
        #self.sigmoid_coef = data['n_dim']  / (n_points)
        #self.variance_leaf_from_root = 2 * np.mean((np.array(self.leaf_means) - self.prior_mean) ** 2)
        # set sd to 3 times the empirical sd so that leaf node means are highly plausible (avoid too much smoothing)
        #self.variance_coef = 1.0 * self.variance_leaf_from_root
        if self.root.is_leaf:
            self.variance_coef = 1.0
        else:
            node_means = np.array(node_means)
            node_parent_means = np.array(node_parent_means)
            node_split_times = np.array(node_split_times)
            node_parent_split_times = np.array(node_parent_split_times)
            tmp_den = sigmoid(self.sigmoid_coef * node_split_times) \
                        - sigmoid(self.sigmoid_coef * node_parent_split_times)
            tmp_num = (node_means - node_parent_means) ** 2
            variance_coef_est = np.mean(tmp_num / tmp_den)
            self.variance_coef = variance_coef_est
            print 'sigmoid_coef = %.3f, variance_coef = %.3f' % (self.sigmoid_coef, variance_coef_est)

    def grow(self, data, settings, param, cache):
        """
        sample a Mondrian tree (each Mondrian block is restricted to range of training data in that block)
        """
        if settings.debug:
            print 'entering grow'
        while self.grow_nodes:
            node_id = self.grow_nodes.pop(0)
            train_ids = self.train_ids[node_id]
            if settings.debug:
                print 'node_id = %s' % node_id
            pause_mondrian = self.pause_mondrian(node_id, settings)
            if settings.debug and pause_mondrian:
                print 'pausing mondrian at node = %s, train_ids = %s' % (node_id, self.train_ids[node_id])
            if pause_mondrian or (node_id.sum_range_d == 0):    # BL: redundant now
                split_cost = np.inf
                self.max_split_costs[node_id] = node_id.budget + 0
                self.split_times[node_id] = np.inf  # FIXME: is this correct? inf or budget?
            else:
                split_cost = random.expovariate(node_id.sum_range_d)
                self.max_split_costs[node_id] = split_cost
                self.split_times[node_id] = split_cost + self.get_parent_split_time(node_id, settings)
            new_budget = node_id.budget - split_cost
            if node_id.budget > split_cost:
                feat_id_chosen = sample_multinomial_scores(node_id.range_d)
                split_chosen = random.uniform(node_id.min_d[feat_id_chosen], \
                                                node_id.max_d[feat_id_chosen])
                (train_ids_left, train_ids_right, cache_tmp) = \
                    compute_left_right_statistics(data, param, cache, train_ids, feat_id_chosen, split_chosen, settings)
                left = MondrianBlock(data, settings, new_budget, node_id, get_data_range(data, train_ids_left))
                right = MondrianBlock(data, settings, new_budget, node_id, get_data_range(data, train_ids_right))
                node_id.left, node_id.right = left, right
                self.grow_nodes.append(left)
                self.grow_nodes.append(right)
                self.train_ids[left] = train_ids_left
                self.train_ids[right] = train_ids_right
                #print 'MondrianTree grow'
                #ipdb.set_trace()
                if settings.optype == 'class':
                    self.counts[left] = cache_tmp['cnt_left_chosen']
                    self.counts[right] = cache_tmp['cnt_right_chosen']
                else:
                    self.sum_y[left] = cache_tmp['sum_y_left']
                    self.sum_y2[left] = cache_tmp['sum_y2_left']
                    self.n_points[left] = cache_tmp['n_points_left']
                    self.sum_y[right] = cache_tmp['sum_y_right']
                    self.sum_y2[right] = cache_tmp['sum_y2_right']
                    self.n_points[right] = cache_tmp['n_points_right']
                self.node_info[node_id] = [feat_id_chosen, split_chosen]
                self.non_leaf_nodes.append(node_id)
                node_id.is_leaf = False
                if not settings.draw_mondrian:
                    self.train_ids.pop(node_id)
            else:
                self.leaf_nodes.append(node_id)     # node_id.is_leaf set to True at init

    def gen_cumulative_split_costs_only(self, settings, data):
        """
        creates node_id.cumulative_split_cost as well as a dictionary self.cumulative_split_costs
        helper function for draw_tree
        """
        self.cumulative_split_costs = {}
        if self.root.is_leaf:
            self.cumulative_split_costs[self.root] = 0.
            remaining = []
        else:
            self.cumulative_split_costs[self.root] = self.max_split_costs[self.root]
            remaining = [self.root.left, self.root.right]
        while True:
            try:
                node_id = remaining.pop(0)
            except IndexError:
                break
            self.cumulative_split_costs[node_id] = self.cumulative_split_costs[node_id.parent] \
                                                    + self.max_split_costs[node_id]
            if not node_id.is_leaf:
                remaining.append(node_id.left)
                remaining.append(node_id.right)

    def gen_node_list(self):
        """
        generates an ordered node_list such that parent appears before children
        useful for updating predictive posteriors
        """
        self.node_list = [self.root]
        i = -1
        while True:
            try:
                i += 1
                node_id = self.node_list[i]
            except IndexError:
                break
            if not node_id.is_leaf:
                self.node_list.extend([node_id.left, node_id.right])

    def predict_class(self, x_test, n_class, param, settings):
        """
        predict new label (for classification tasks)
        """
        pred_prob = np.zeros((x_test.shape[0], n_class))
        prob_not_separated_yet = np.ones(x_test.shape[0])
        prob_separated = np.zeros(x_test.shape[0])
        node_list = [self.root]
        d_idx_test = {self.root: np.arange(x_test.shape[0])}
        while True:
            try:
                node_id = node_list.pop(0)
            except IndexError:
                break
            idx_test = d_idx_test[node_id]
            if len(idx_test) == 0:
                continue
            x = x_test[idx_test, :]
            expo_parameter = np.maximum(0, node_id.min_d - x).sum(1) + np.maximum(0, x - node_id.max_d).sum(1)
            prob_not_separated_now = np.exp(-expo_parameter * self.max_split_costs[node_id])
            prob_separated_now = 1 - prob_not_separated_now
            if math.isinf(self.max_split_costs[node_id]):
                # rare scenario where test point overlaps exactly with a training data point
                idx_zero = expo_parameter == 0
                # to prevent nan in computation above when test point overlaps with training data point
                prob_not_separated_now[idx_zero] = 1.
                prob_separated_now[idx_zero] = 0.
            # predictions for idx_test_zero
            # data dependent discounting (depending on how far test data point is from the mondrian block)
            idx_non_zero = expo_parameter > 0
            idx_test_non_zero = idx_test[idx_non_zero]
            expo_parameter_non_zero = expo_parameter[idx_non_zero]
            base = self.get_prior_mean(node_id, param, settings)
            if np.any(idx_non_zero):
                num_tables_k, num_customers, num_tables = self.get_counts(self.cnt[node_id])
                # expected discount (averaging over time of cut which is a truncated exponential)
                # discount = (expo_parameter_non_zero / (expo_parameter_non_zero + settings.discount_param)) * \
                #       (-np.expm1(-(expo_parameter_non_zero + settings.discount_param) * self.max_split_costs[node_id]))
                discount = (expo_parameter_non_zero / (expo_parameter_non_zero + settings.discount_param)) \
                    * (-np.expm1(-(expo_parameter_non_zero + settings.discount_param) * self.max_split_costs[node_id])) \
                    / (-np.expm1(-expo_parameter_non_zero * self.max_split_costs[node_id]))
                discount_per_num_customers = discount / num_customers
                pred_prob_tmp = num_tables * discount_per_num_customers[:, np.newaxis] * base \
                        + self.cnt[node_id] / num_customers - discount_per_num_customers[:, np.newaxis] * num_tables_k
                pred_prob[idx_test_non_zero, :] += prob_separated_now[idx_non_zero][:, np.newaxis] \
                                            * prob_not_separated_yet[idx_test_non_zero][:, np.newaxis] * pred_prob_tmp
                prob_not_separated_yet[idx_test] *= prob_not_separated_now
            # predictions for idx_test_zero
            if math.isinf(self.max_split_costs[node_id]) and np.any(idx_zero):
                idx_test_zero = idx_test[idx_zero]
                pred_prob_node_id = self.compute_posterior_mean_normalized_stable(self.cnt[node_id], \
                                            self.get_discount_node_id(node_id, settings), base, settings)
                pred_prob[idx_test_zero, :] += prob_not_separated_yet[idx_test_zero][:, np.newaxis] * pred_prob_node_id
            try:
                feat_id, split = self.node_info[node_id]
                cond = x[:, feat_id] <= split
                left, right = get_children_id(node_id)
                d_idx_test[left], d_idx_test[right] = idx_test[cond], idx_test[~cond]
                node_list.append(left)
                node_list.append(right)
            except KeyError:
                pass
        if True or settings.debug:
            check_if_zero(np.sum(np.abs(np.sum(pred_prob, 1) - 1)))
        return pred_prob

    def predict_real(self, x_test, y_test, param, settings):
        """
        predict new label (for regression tasks)
        """
        pred_mean = np.zeros(x_test.shape[0])
        pred_second_moment = np.zeros(x_test.shape[0])
        pred_sample = np.zeros(x_test.shape[0])
        log_pred_prob = -np.inf * np.ones(x_test.shape[0])
        prob_not_separated_yet = np.ones(x_test.shape[0])
        prob_separated = np.zeros(x_test.shape[0])
        node_list = [self.root]
        d_idx_test = {self.root: np.arange(x_test.shape[0])}
        while True:
            try:
                node_id = node_list.pop(0)
            except IndexError:
                break
            idx_test = d_idx_test[node_id]
            if len(idx_test) == 0:
                continue
            x = x_test[idx_test, :]
            expo_parameter = np.maximum(0, node_id.min_d - x).sum(1) + np.maximum(0, x - node_id.max_d).sum(1)
            prob_not_separated_now = np.exp(-expo_parameter * self.max_split_costs[node_id])
            prob_separated_now = 1 - prob_not_separated_now
            if math.isinf(self.max_split_costs[node_id]):
                # rare scenario where test point overlaps exactly with a training data point
                idx_zero = expo_parameter == 0
                # to prevent nan in computation above when test point overlaps with training data point
                prob_not_separated_now[idx_zero] = 1.
                prob_separated_now[idx_zero] = 0.
            # predictions for idx_test_zero
            idx_non_zero = expo_parameter > 0
            idx_test_non_zero = idx_test[idx_non_zero]
            n_test_non_zero = len(idx_test_non_zero)
            expo_parameter_non_zero = expo_parameter[idx_non_zero]
            if np.any(idx_non_zero):
                # expected variance (averaging over time of cut which is a truncated exponential)
                # NOTE: expected variance is approximate since E[f(x)] not equal to f(E[x])
                expected_cut_time = 1.0 / expo_parameter_non_zero
                if not np.isinf(self.max_split_costs[node_id]):
                    tmp_exp_term_arg = -self.max_split_costs[node_id] * expo_parameter_non_zero
                    tmp_exp_term = np.exp(tmp_exp_term_arg)
                    expected_cut_time -= self.max_split_costs[node_id] * tmp_exp_term / (-np.expm1(tmp_exp_term_arg))
                try:
                    assert np.all(expected_cut_time >= 0.)
                except AssertionError:
                    print tmp_exp_term_arg
                    print tmp_exp_term
                    print expected_cut_time
                    print np.any(np.isnan(expected_cut_time))
                    print 1.0 / expo_parameter_non_zero
                    raise AssertionError
                if not settings.smooth_hierarchically:
                    pred_mean_tmp = self.sum_y[node_id] / float(self.n_points[node_id])
                    pred_second_moment_tmp = self.sum_y2[node_id] / float(self.n_points[node_id]) + param.noise_variance
                else:
                    pred_mean_tmp, pred_second_moment_tmp = self.pred_moments[node_id]
                    # FIXME: approximate since E[f(x)] not equal to f(E[x])
                    expected_split_time = expected_cut_time + self.get_parent_split_time(node_id, settings)
                    variance_from_mean = self.variance_coef * (sigmoid(self.sigmoid_coef * expected_split_time) \
                                        - sigmoid(self.sigmoid_coef * self.get_parent_split_time(node_id, settings)))
                    pred_second_moment_tmp += variance_from_mean
                pred_variance_tmp = pred_second_moment_tmp - pred_mean_tmp ** 2
                pred_sample_tmp = pred_mean_tmp + np.random.randn(n_test_non_zero) * np.sqrt(pred_variance_tmp)
                log_pred_prob_tmp = compute_gaussian_logpdf(pred_mean_tmp, pred_variance_tmp, y_test[idx_test_non_zero])
                prob_separated_now_weighted = \
                        prob_separated_now[idx_non_zero] * prob_not_separated_yet[idx_test_non_zero]
                pred_mean[idx_test_non_zero] += prob_separated_now_weighted * pred_mean_tmp
                pred_sample[idx_test_non_zero] += prob_separated_now_weighted * pred_sample_tmp
                pred_second_moment[idx_test_non_zero] += prob_separated_now_weighted * pred_second_moment_tmp
                log_pred_prob[idx_test_non_zero] = logsumexp_array(log_pred_prob[idx_test_non_zero], \
                                                    np.log(prob_separated_now_weighted) + log_pred_prob_tmp)
                prob_not_separated_yet[idx_test] *= prob_not_separated_now
            # predictions for idx_test_zero
            if math.isinf(self.max_split_costs[node_id]) and np.any(idx_zero):
                idx_test_zero = idx_test[idx_zero]
                n_test_zero = len(idx_test_zero)
                if not settings.smooth_hierarchically:
                    pred_mean_node_id = self.sum_y[node_id] / float(self.n_points[node_id])
                    pred_second_moment_node_id = self.sum_y2[node_id] / float(self.n_points[node_id]) \
                                                    + param.noise_variance
                else:
                    pred_mean_node_id, pred_second_moment_node_id = self.pred_moments[node_id]
                pred_variance_node_id = pred_second_moment_node_id - pred_mean_node_id ** 2
                pred_sample_node_id = pred_mean_node_id + np.random.randn(n_test_zero) * np.sqrt(pred_variance_node_id)
                log_pred_prob_node_id = \
                        compute_gaussian_logpdf(pred_mean_node_id, pred_variance_node_id, y_test[idx_test_zero])
                pred_mean[idx_test_zero] += prob_not_separated_yet[idx_test_zero] * pred_mean_node_id
                pred_sample[idx_test_zero] += prob_not_separated_yet[idx_test_zero] * pred_sample_node_id
                pred_second_moment[idx_test_zero] += prob_not_separated_yet[idx_test_zero] * pred_second_moment_node_id
                log_pred_prob[idx_test_zero] = logsumexp_array(log_pred_prob[idx_test_zero], \
                                                np.log(prob_not_separated_yet[idx_test_zero]) + log_pred_prob_node_id)
            try:
                feat_id, split = self.node_info[node_id]
                cond = x[:, feat_id] <= split
                left, right = get_children_id(node_id)
                d_idx_test[left], d_idx_test[right] = idx_test[cond], idx_test[~cond]
                node_list.append(left)
                node_list.append(right)
            except KeyError:
                pass
        pred_var = pred_second_moment - (pred_mean ** 2)
        if True or settings.debug:  # FIXME: remove later
            assert not np.any(np.isnan(pred_mean))
            assert not np.any(np.isnan(pred_var))
            try:
                assert np.all(pred_var >= 0.)
            except AssertionError:
                min_pred_var = np.min(pred_var)
                print 'min_pred_var = %s' % min_pred_var
                assert np.abs(min_pred_var) < 1e-3  # allowing some numerical errors
            assert not np.any(np.isnan(log_pred_prob))
        return (pred_mean, pred_var, pred_second_moment, log_pred_prob, pred_sample)

    def extend_mondrian(self, data, train_ids_new, settings, param, cache):
        """
        extend Mondrian tree to include new training data indexed by train_ids_new
        """
        self.extend_mondrian_block(self.root, train_ids_new, data, settings, param, cache)
        if settings.debug:
            print 'completed extend_mondrian'
            self.check_tree(settings, data)

    def check_tree(self, settings, data):
        """
        check if tree violates any sanity check
        """
        if settings.debug:
            #print '\nchecking tree'
            print '\nchecking tree: printing tree first'
            self.print_tree(settings)
        for node_id in self.non_leaf_nodes:
            assert node_id.left.parent == node_id.right.parent == node_id
            assert not node_id.is_leaf
            if settings.optype == 'class':
                assert np.count_nonzero(self.counts[node_id]) > 1
            assert not self.pause_mondrian(node_id, settings)
            if node_id != self.root:
                assert np.all(node_id.min_d >= node_id.parent.min_d)
                assert np.all(node_id.max_d <= node_id.parent.max_d)
            if settings.optype == 'class':
                try:
                    check_if_zero(np.sum(np.abs(self.counts[node_id] - \
                            self.counts[node_id.left] - self.counts[node_id.right])))
                except AssertionError:
                    print 'counts: node = %s, left = %s, right = %s' \
                            % (self.counts[node_id], self.counts[node_id.left], self.counts[node_id.right])
                    raise AssertionError
            if settings.budget == -1:
                assert math.isinf(node_id.budget)
            check_if_zero(self.split_times[node_id] - self.get_parent_split_time(node_id, settings) \
                    - self.max_split_costs[node_id])
        if settings.optype == 'class':
            num_data_points = 0
        for node_id in self.leaf_nodes:
            assert node_id.is_leaf
            assert math.isinf(self.max_split_costs[node_id])
            if settings.budget == -1:
                assert math.isinf(node_id.budget)
            if settings.optype == 'class':
                num_data_points += self.counts[node_id].sum()
                assert np.count_nonzero(self.counts[node_id]) == 1
                assert self.pause_mondrian(node_id, settings)
            if node_id != self.root:
                assert np.all(node_id.min_d >= node_id.parent.min_d)
                assert np.all(node_id.max_d <= node_id.parent.max_d)
        if settings.optype == 'class' and settings.debug:
            print 'num_train = %s, number of data points at leaf nodes = %s' % \
                    (data['n_train'], num_data_points)
        set_non_leaf = set(self.non_leaf_nodes)
        set_leaf = set(self.leaf_nodes)
        assert (set_leaf & set_non_leaf) == set([])
        assert set_non_leaf == set(self.node_info.keys())
        assert len(set_leaf) == len(self.leaf_nodes)
        assert len(set_non_leaf) == len(self.non_leaf_nodes)

    def extend_mondrian_block(self, node_id, train_ids_new, data, settings, param, cache):
        """
        conditional Mondrian algorithm that extends a Mondrian block to include new training data
        """
        if settings.debug:
            print 'entered extend_mondrian_block'
            print '\nextend_mondrian_block: node_id = %s' % node_id
        if not train_ids_new.size:
            if settings.debug:
                print 'nothing to extend here; train_ids_new = %s' % train_ids_new
            # nothing to extend
            return
        min_d, max_d = get_data_min_max(data, train_ids_new)
        additional_extent_lower = np.maximum(0, node_id.min_d - min_d)
        additional_extent_upper = np.maximum(0, max_d - node_id.max_d)
        expo_parameter = float(additional_extent_lower.sum() + additional_extent_upper.sum())
        if expo_parameter == 0:
            split_cost = np.inf
        else:
            split_cost = random.expovariate(expo_parameter)     # will be updated below in case mondrian is paused
        unpause_paused_mondrian = False
        if settings.debug:
            print 'is_leaf = %s, pause_mondrian = %s, sum_range_d = %s' % \
                    (node_id.is_leaf, self.pause_mondrian(node_id, settings), node_id.sum_range_d)
        if self.pause_mondrian(node_id, settings):
            assert node_id.is_leaf
            split_cost = np.inf
            if settings.optype == 'class':
                y_unique = np.unique(data['y_train'][train_ids_new])
                # FIXME: node_id.sum_range_d not tested
                unpause_paused_mondrian = not( (len(y_unique) == 1) and (self.counts[node_id][y_unique] > 0) )
            else:
                n_points_new = len(data['y_train'][train_ids_new])
                unpause_paused_mondrian = \
                        not( (n_points_new + self.n_points[node_id]) < settings.min_samples_split )
                        # node_id.sum_range_d not tested
            if settings.debug:
                print 'trying to extend a paused Mondrian; is_leaf = %s, node_id = %s' % (node_id.is_leaf, node_id)
                if settings.optype == 'class':
                    print 'y_unique = %s, counts = %s, split_cost = %s, max_split_costs = %s' % \
                        (y_unique, self.counts[node_id], split_cost, self.max_split_costs[node_id])
        if split_cost >= self.max_split_costs[node_id]:
            # take root form of node_id (no cut outside the extent of the current block)
            if not node_id.is_leaf:
                if settings.debug:
                    print 'take root form: non-leaf node'
                feat_id, split = self.node_info[node_id]
                update_range_stats(node_id, (min_d, max_d)) # required here as well
                left, right = node_id.left, node_id.right
                cond = data['x_train'][train_ids_new, feat_id] <= split
                train_ids_new_left, train_ids_new_right = train_ids_new[cond], train_ids_new[~cond]
                self.add_training_points_to_node(node_id, train_ids_new, data, param, settings, cache, False)
                self.extend_mondrian_block(left, train_ids_new_left, data, settings, param, cache)
                self.extend_mondrian_block(right, train_ids_new_right, data, settings, param, cache)
            else:
                # reached a leaf; add train_ids_new to node_id & update range
                if settings.debug:
                    print 'take root form: leaf node'
                assert node_id.is_leaf
                update_range_stats(node_id, (min_d, max_d))
                self.add_training_points_to_node(node_id, train_ids_new, data, param, settings, cache, True)
                # FIXME: node_id.sum_range_d tested here; perhaps move this to pause_mondrian?
                unpause_paused_mondrian = unpause_paused_mondrian and (node_id.sum_range_d != 0)
                if not self.pause_mondrian(node_id, settings):
                    assert unpause_paused_mondrian
                    self.leaf_nodes.remove(node_id)
                    self.grow_nodes = [node_id]
                    self.grow(data, settings, param, cache)
        else:
            # initialize "outer mondrian"
            if settings.debug:
                print 'trying to introduce a cut outside current block'
            new_block = MondrianBlock(data, settings, node_id.budget, node_id.parent, \
                        get_data_range_from_min_max(np.minimum(min_d, node_id.min_d), np.maximum(max_d, node_id.max_d)))
            init_update_posterior_node_incremental(self, data, param, settings, cache, new_block, \
                    train_ids_new, node_id)      # counts of outer block are initialized with counts of current block
            if node_id.is_leaf:
                warn('\nWARNING: a leaf should not be expanded here; printing out some diagnostics')
                print 'node_id = %s, is_leaf = %s, max_split_cost = %s, split_cost = %s' \
                        % (node_id, node_id.is_leaf, self.max_split_costs[node_id], split_cost)
                print 'counts = %s\nmin_d = \n%s\nmax_d = \n%s' % (self.counts[node_id], node_id.min_d, node_id.max_d)
                raise Exception('a leaf should be expanded via grow call; see diagnostics above')
            if settings.debug:
                print 'looks like cut possible'
            # there is a cut outside the extent of the current block
            feat_score = additional_extent_lower + additional_extent_upper
            feat_id = sample_multinomial_scores(feat_score)
            draw_from_lower = np.random.rand() <= (additional_extent_lower[feat_id] / feat_score[feat_id])
            if draw_from_lower:
                split = random.uniform(min_d[feat_id], node_id.min_d[feat_id])
            else:
                split = random.uniform(node_id.max_d[feat_id], max_d[feat_id])
            assert (split < node_id.min_d[feat_id]) or (split > node_id.max_d[feat_id])
            new_budget = node_id.budget - split_cost
            cond = data['x_train'][train_ids_new, feat_id] <= split
            train_ids_new_left, train_ids_new_right = train_ids_new[cond], train_ids_new[~cond]
            is_left = split > node_id.max_d[feat_id]    # is existing block the left child of "outer mondrian"?
            if is_left:
                train_ids_new_child = train_ids_new_right   # new_child is the other child of "outer mondrian"
            else:
                train_ids_new_child = train_ids_new_left
            # grow the "unconditional mondrian child" of the "outer mondrian"
            new_child = MondrianBlock(data, settings, new_budget, new_block, get_data_range(data, train_ids_new_child))
            if settings.debug:
                print 'new_block = %s' % new_block
                print 'new_child = %s' % new_child
            self.train_ids[new_child] = train_ids_new_child     # required for grow call below
            init_update_posterior_node_incremental(self, data, param, settings, cache, new_child, train_ids_new_child)
            self.node_info[new_block] = (feat_id, split)
            if settings.draw_mondrian:
                train_ids_new_block = np.append(self.train_ids[node_id], train_ids_new)
                self.train_ids[new_block] = train_ids_new_block
            self.non_leaf_nodes.append(new_block)
            new_block.is_leaf = False
            # update budget and call the "conditional mondrian child" of the "outer mondrian"
            node_id.budget = new_budget
            # self.max_split_costs[new_child] will be added in the grow call above
            self.max_split_costs[new_block] = split_cost
            self.split_times[new_block] = split_cost + self.get_parent_split_time(node_id, settings)
            self.max_split_costs[node_id] -= split_cost
            check_if_zero(self.split_times[node_id] - self.split_times[new_block] - self.max_split_costs[node_id])
            # grow the new child of the "outer mondrian"
            self.grow_nodes = [new_child]
            self.grow(data, settings, param, cache)
            # update tree structure and extend "conditional mondrian child" of the "outer mondrian"
            if node_id == self.root:
                self.root = new_block
            else:
                if settings.debug:
                    assert (node_id.parent.left == node_id) or (node_id.parent.right == node_id)
                if node_id.parent.left == node_id:
                    node_id.parent.left = new_block
                else:
                    node_id.parent.right = new_block
            node_id.parent = new_block
            if is_left:
                new_block.left = node_id
                new_block.right = new_child
                self.extend_mondrian_block(node_id, train_ids_new_left, data, settings, param, cache)
            else:
                new_block.left = new_child
                new_block.right = node_id
                self.extend_mondrian_block(node_id, train_ids_new_right, data, settings, param, cache)

    def add_training_points_to_node(self, node_id, train_ids_new, data, param, settings, cache, pause_mondrian=False):
        """
        add a training data point to a node in the tree
        """
        # range updated in extend_mondrian_block
        if settings.draw_mondrian or pause_mondrian:
            self.train_ids[node_id] = np.append(self.train_ids[node_id], train_ids_new)
        update_posterior_node_incremental(self, data, param, settings, cache, node_id, train_ids_new)

    def update_posterior_counts(self, param, data, settings):
        """
        posterior update for hierarchical normalized stable distribution
        using interpolated Kneser Ney smoothing (where number of tables serving a dish at a restaurant is atmost 1)
        NOTE: implementation optimized for minibatch training where more than one data point added per minibatch
        if only 1 datapoint is added, lots of counts will be unnecesarily updated
        """
        self.cnt = {}
        node_list = [self.root]
        while True:
            try:
                node_id = node_list.pop(0)
            except IndexError:
                break
            if node_id.is_leaf:
                cnt = self.counts[node_id]
            else:
                cnt = np.minimum(self.counts[node_id.left], 1) + np.minimum(self.counts[node_id.right], 1)
                node_list.extend([node_id.left, node_id.right])
            self.cnt[node_id] = cnt

    def update_predictive_posteriors(self, param, data, settings):
        """
        update predictive posterior for hierarchical normalized stable distribution
        pred_prob computes posterior mean of the label distribution at each node recursively
        """
        node_list = [self.root]
        if settings.debug:
            self.gen_node_ids_print()
        while True:
            try:
                node_id = node_list.pop(0)
            except IndexError:
                break
            base = self.get_prior_mean(node_id, param, settings)
            discount = self.get_discount_node_id(node_id, settings)
            cnt = self.cnt[node_id]
            if not node_id.is_leaf:
                self.pred_prob[node_id] = self.compute_posterior_mean_normalized_stable(cnt, discount, base, settings)
                node_list.extend([node_id.left, node_id.right])
            if settings.debug and False:
                print 'node_id = %20s, is_leaf = %5s, discount = %.2f, cnt = %s, base = %s, pred_prob = %s' \
                        % (self.node_ids_print[node_id], node_id.is_leaf, discount, cnt, base, self.pred_prob[node_id])

    def get_variance_node(self, node_id, param, settings):
        # the non-linear transformation should be a monotonically non-decreasing function
        # if the function saturates (e.g. sigmoid) children will be closer to parent deeper down the tree
        # var = self.variance_coef * (sigmoid(self.sigmoid_coef * self.split_times[node_id]) \
        #        - sigmoid(self.sigmoid_coef * self.get_parent_split_time(node_id, settings)))
        var = self.variance_coef * (sigmoid(self.sigmoid_coef * self.split_times[node_id]) \
                - sigmoid(self.sigmoid_coef * self.get_parent_split_time(node_id, settings)))
        return var

    def update_posterior_gaussians(self, param, data, settings):
        """
        computes marginal gaussian distribution at each node of the tree using gaussian belief propagation
        the solution is exact since underlying graph is a tree
        solution takes O(#nodes) time, which is much more efficient than naive GP implementation which
        would cost O(#nodes^3) time
        """
        self.gen_node_list()
        self.message_to_parent = {}
        self.message_from_parent = {}
        self.likelihood_children = {}
        self.pred_param = {}
        self.pred_moments = {}
        for node_id in self.node_list[::-1]:
            if node_id.is_leaf:
                # use marginal likelihood of data at this leaf
                mean = self.sum_y[node_id] / float(self.n_points[node_id])
                variance = self.get_variance_node(node_id, param, settings) \
                            + self.noise_variance / float(self.n_points[node_id])
                precision = 1.0 / variance
                self.message_to_parent[node_id] = np.array([mean, precision])
                self.likelihood_children[node_id] = np.array([mean, self.noise_precision*float(self.n_points[node_id])])
            else:
                likelihood_children = multiply_gaussians(self.message_to_parent[node_id.left], \
                                                    self.message_to_parent[node_id.right])
                mean = likelihood_children[0]
                self.likelihood_children[node_id] = likelihood_children
                variance = self.get_variance_node(node_id, param, settings) + 1.0 / likelihood_children[1]
                precision = 1.0 / variance
                self.message_to_parent[node_id] = np.array([mean, precision])
        variance_at_root = self.get_variance_node(node_id, param, settings)
        self.message_from_parent[self.root] = np.array([param.prior_mean, variance_at_root])
        for node_id in self.node_list:
            # pred_param stores the mean and precision
            self.pred_param[node_id] = multiply_gaussians(self.message_from_parent[node_id], \
                                            self.likelihood_children[node_id])
            # pred_moments stores the first and second moments (useful for prediction)
            self.pred_moments[node_id] = np.array([self.pred_param[node_id][0], \
                    1.0 / self.pred_param[node_id][1] + self.pred_param[node_id][0] ** 2 + self.noise_variance])
            if not node_id.is_leaf:
                self.message_from_parent[node_id.left] = \
                        multiply_gaussians(self.message_from_parent[node_id], self.message_to_parent[node_id.right])
                self.message_from_parent[node_id.right] = \
                        multiply_gaussians(self.message_from_parent[node_id], self.message_to_parent[node_id.left])

    def update_posterior_counts_and_predictive_posteriors(self, param, data, settings):
        if settings.optype == 'class':
            # update posterior counts
            self.update_posterior_counts(param, data, settings)
            # update predictive posteriors
            self.update_predictive_posteriors(param, data, settings)
        else:
            # updates hyperparameters in param (common to all trees)
            self.update_gaussian_hyperparameters(param, data, settings)
            # updates hyperparameters in self (independent for each tree)
            # self.update_gaussian_hyperparameters_indep(param, data, settings)
            if settings.smooth_hierarchically:
                self.update_posterior_gaussians(param, data, settings)

    def get_prior_mean(self, node_id, param, settings):
        if settings.optype == 'class':
            if node_id == self.root:
                base = param.base_measure
            else:
                base = self.pred_prob[node_id.parent]
        else:
            base = None     # for settings.settings.smooth_hierarchically = False
        return base

    def get_discount_node_id(self, node_id, settings):
        """
        compute discount for a node (function of discount_param, time of split and time of split of parent)
        """
        discount = math.exp(-settings.discount_param * self.max_split_costs[node_id])
        return discount

    def compute_posterior_mean_normalized_stable(self, cnt, discount, base, settings):
        num_tables_k, num_customers, num_tables = self.get_counts(cnt)
        pred_prob = (cnt - discount * num_tables_k + discount * num_tables * base) / num_customers
        if settings.debug:
            check_if_one(pred_prob.sum())
        return pred_prob

    def get_counts(self, cnt):
        num_tables_k = np.minimum(cnt, 1)
        num_customers = float(cnt.sum())
        num_tables = float(num_tables_k.sum())
        return (num_tables_k, num_customers, num_tables)


def get_data_range(data, train_ids):
    """
    returns min, max, range and linear dimension of training data
    """
    min_d, max_d = get_data_min_max(data, train_ids)
    range_d = max_d - min_d
    sum_range_d = float(range_d.sum())
    return (min_d, max_d, range_d, sum_range_d)


def get_data_min_max(data, train_ids):
    """
    returns min, max of training data
    """
    x_tmp = data['x_train'].take(train_ids, 0)
    min_d = np.min(x_tmp, 0)
    max_d = np.max(x_tmp, 0)
    return (min_d, max_d)


def get_data_range_from_min_max(min_d, max_d):
    range_d = max_d - min_d
    sum_range_d = float(range_d.sum())
    return (min_d, max_d, range_d, sum_range_d)


def update_range_stats(node_id, (min_d, max_d)):
    """
    updates min and max of training data at this block
    """
    node_id.min_d = np.minimum(node_id.min_d, min_d)
    node_id.max_d = np.maximum(node_id.max_d, max_d)
    node_id.range_d = node_id.max_d - node_id.min_d
    node_id.sum_range_d = float(node_id.range_d.sum())


def get_children_id(parent):
    return (parent.left, parent.right)


class MondrianForest(Forest):
    """
    defines Mondrian forest
    variables:
    - forest     : stores the Mondrian forest
    methods:
    - fit(data, train_ids_current_minibatch, settings, param, cache)            : batch training
    - partial_fit(data, train_ids_current_minibatch, settings, param, cache)    : online training
    - evaluate_predictions (see Forest in mondrianforest_utils.py)              : predictions
    """
    def __init__(self, settings, data):
    #def __init__(self, settings, n_dim):
        self.forest = [None] * settings.n_mondrians
        if settings.optype == 'class':
            settings.discount_param = settings.discount_factor * data['n_dim']
            #settings.discount_param = settings.discount_factor * n_dim

    def fit(self, data, train_ids_current_minibatch, settings, param, cache):
        for i_t, tree in enumerate(self.forest):
            if settings.verbose >= 2 or settings.debug:
                print 'tree_id = %s' % i_t
            tree = self.forest[i_t] = MondrianTree(data, train_ids_current_minibatch, settings, param, cache)
            tree.update_posterior_counts_and_predictive_posteriors(param, data, settings)
            tree.check_tree(settings, data)

    def partial_fit(self, data, train_ids_current_minibatch, settings, param, cache):
        for i_t, tree in enumerate(self.forest):
            if settings.verbose >= 2 or settings.debug:
                print 'tree_id = %s' % i_t
            tree.extend_mondrian(data, train_ids_current_minibatch, settings, param, cache)
            tree.update_posterior_counts_and_predictive_posteriors(param, data, settings)

    def print_forest(self, settings):
        for i_t, tree in enumerate(self.forest):
            print '-'*100
            tree.print_tree(settings)

#def main():
#    time_0 = time.clock()
#    settings = process_command_line()
#    print
#    print '%' * 120
#    print 'Beginning mondrianforest.py'
#    print 'Current settings:'
#    pp.pprint(vars(settings))
#
#    # Resetting random seed
#    reset_random_seed(settings)
#
#    # Loading data
#    print '\nLoading data ...'
#    data = load_data(settings)
#    print 'Loading data ... completed'
#    print 'Dataset name = %s' % settings.dataset
#    print 'Characteristics of the dataset:'
#    print 'n_train = %d, n_test = %d, n_dim = %d' %\
#            (data['n_train'], data['n_test'], data['n_dim'])
#    if settings.optype == 'class':
#        print 'n_class = %d' % (data['n_class'])
#
#    # precomputation
#    param, cache = precompute_minimal(data, settings)
#    time_init = time.clock() - time_0
#
#    print '\nCreating Mondrian forest'
#    # online training with minibatches
#    time_method_sans_init = 0.
#    time_prediction = 0.
#    mf = MondrianForest(settings, data)
#    if settings.store_every:
#        log_prob_test_minibatch = -np.inf * np.ones(settings.n_minibatches)
#        log_prob_train_minibatch = -np.inf * np.ones(settings.n_minibatches)
#        metric_test_minibatch = -np.inf * np.ones(settings.n_minibatches)
#        metric_train_minibatch = -np.inf * np.ones(settings.n_minibatches)
#        time_method_minibatch = np.inf * np.ones(settings.n_minibatches)
#        forest_numleaves_minibatch = np.zeros(settings.n_minibatches)
#    for idx_minibatch in range(settings.n_minibatches):
#        time_method_init = time.clock()
#        is_last_minibatch = (idx_minibatch == settings.n_minibatches - 1)
#        print_results = is_last_minibatch or (settings.verbose >= 2) or settings.debug
#        if print_results:
#            print '*' * 120
#            print 'idx_minibatch = %5d' % idx_minibatch
#        train_ids_current_minibatch = data['train_ids_partition']['current'][idx_minibatch]
#        if settings.debug:
#            print 'bagging = %s, train_ids_current_minibatch = %s' % \
#                    (settings.bagging, train_ids_current_minibatch)
#        if idx_minibatch == 0:
#            mf.fit(data, train_ids_current_minibatch, settings, param, cache)
#        else:
#            mf.partial_fit(data, train_ids_current_minibatch, settings, param, cache)
#        for i_t, tree in enumerate(mf.forest):
#            if settings.debug or settings.verbose >= 2:
#                print '-'*100
#                tree.print_tree(settings)
#                print '.'*100
#            if settings.draw_mondrian:
#                tree.draw_mondrian(data, settings, idx_minibatch, i_t)
#                if settings.save == 1:
#                    filename_plot = get_filename_mf(settings)[:-2]
#                    if settings.store_every:
#                        plt.savefig(filename_plot + '-mondrians_minibatch-' + str(idx_minibatch) + '.pdf', format='pdf')
#        time_method_sans_init += time.clock() - time_method_init
#        time_method = time_method_sans_init + time_init
#
#        # Evaluate
#        if is_last_minibatch or settings.store_every:
#            time_predictions_init = time.clock()
#            weights_prediction = np.ones(settings.n_mondrians) * 1.0 / settings.n_mondrians
#            if False:
#                if print_results:
#                    print 'Results on training data (log predictive prob is bogus)'
#                train_ids_cumulative = data['train_ids_partition']['cumulative'][idx_minibatch]
#                # NOTE: some of these data points are not used for "training" if bagging is used
#                pred_forest_train, metrics_train = \
#                        mf.evaluate_predictions(data, data['x_train'][train_ids_cumulative, :], \
#                        data['y_train'][train_ids_cumulative], \
#                        settings, param, weights_prediction, print_results)
#            else:
#                # not computing metrics on training data
#                metrics_train = {'log_prob': -np.inf, 'acc': 0, 'mse': np.inf}
#                pred_forest_train = None
#            if print_results:
#                print '\nResults on test data'
#            pred_forest_test, metrics_test = \
#                mf.evaluate_predictions(data, data['x_test'], data['y_test'], \
#                settings, param, weights_prediction, print_results)
#            name_metric = settings.name_metric     # acc or mse
#            log_prob_train = metrics_train['log_prob']
#            log_prob_test = metrics_test['log_prob']
#            metric_train = metrics_train[name_metric]
#            metric_test = metrics_test[name_metric]
#            if settings.store_every:
#                log_prob_train_minibatch[idx_minibatch] = metrics_train['log_prob']
#                log_prob_test_minibatch[idx_minibatch] = metrics_test['log_prob']
#                metric_train_minibatch[idx_minibatch] = metrics_train[name_metric]
#                metric_test_minibatch[idx_minibatch] = metrics_test[name_metric]
#                time_method_minibatch[idx_minibatch] = time_method
#                tree_numleaves = np.zeros(settings.n_mondrians)
#                for i_t, tree in enumerate(mf.forest):
#                    tree_numleaves[i_t] = len(tree.leaf_nodes)
#                forest_numleaves_minibatch[idx_minibatch] = np.mean(tree_numleaves)
#            time_prediction += time.clock() - time_predictions_init
#
#    # printing test performance:
#    if settings.store_every:
#        print 'printing test performance for every minibatch:'
#        print 'idx_minibatch\tmetric_test\ttime_method\tnum_leaves'
#        for idx_minibatch in range(settings.n_minibatches):
#            print '%10d\t%.3f\t\t%.3f\t\t%.1f' % \
#                    (idx_minibatch, \
#                    metric_test_minibatch[idx_minibatch], \
#                    time_method_minibatch[idx_minibatch], forest_numleaves_minibatch[idx_minibatch])
#    print '\nFinal forest stats:'
#    tree_stats = np.zeros((settings.n_mondrians, 2))
#    tree_average_depth = np.zeros(settings.n_mondrians)
#    for i_t, tree in enumerate(mf.forest):
#        tree_stats[i_t, -2:] = np.array([len(tree.leaf_nodes), len(tree.non_leaf_nodes)])
#        tree_average_depth[i_t] = tree.get_average_depth(settings, data)[0]
#    print 'mean(num_leaves) = %.1f, mean(num_non_leaves) = %.1f, mean(tree_average_depth) = %.1f' \
#            % (np.mean(tree_stats[:, -2]), np.mean(tree_stats[:, -1]), np.mean(tree_average_depth))
#    print 'n_train = %d, log_2(n_train) = %.1f, mean(tree_average_depth) = %.1f +- %.1f' \
#            % (data['n_train'], np.log2(data['n_train']), np.mean(tree_average_depth), np.std(tree_average_depth))
#
#    if settings.draw_mondrian:
#        if settings.save == 1:
#            plt.savefig(filename_plot + '-mondrians-final.pdf', format='pdf')
#        else:
#            plt.show()
#
#    # Write results to disk (timing doesn't include saving)
#    time_total = time.clock() - time_0
#    # resetting
#    if settings.save == 1:
#        filename = get_filename_mf(settings)
#        print 'filename = ' + filename
#        results = {'log_prob_test': log_prob_test, 'log_prob_train': log_prob_train, \
#                    'metric_test': metric_test, 'metric_train': metric_train, \
#                'time_total': time_total, 'time_method': time_method, \
#                'time_init': time_init, 'time_method_sans_init': time_method_sans_init,\
#                'time_prediction': time_prediction}
#        if 'log_prob2' in metrics_test:
#            results['log_prob2_test'] = metrics_test['log_prob2']
#        store_data = settings.dataset[:3] == 'toy' or settings.dataset == 'sim-reg'
#        if store_data:
#            results['data'] = data
#        if settings.store_every:
#            results['log_prob_test_minibatch'] = log_prob_test_minibatch
#            results['log_prob_train_minibatch'] = log_prob_train_minibatch
#            results['metric_test_minibatch'] = metric_test_minibatch
#            results['metric_train_minibatch'] = metric_train_minibatch
#            results['time_method_minibatch'] = time_method_minibatch
#            results['forest_numleaves_minibatch'] = forest_numleaves_minibatch
#        results['settings'] = settings
#        results['tree_stats'] = tree_stats
#        results['tree_average_depth'] = tree_average_depth
#        pickle.dump(results, open(filename, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
#        # storing final predictions as well; recreate new "results" dict
#        results = {'pred_forest_train': pred_forest_train, \
#                    'pred_forest_test': pred_forest_test}
#        filename2 = filename[:-2] + '.tree_predictions.p'
#        pickle.dump(results, open(filename2, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
#
#    time_total = time.clock() - time_0
#    print
#    print 'Time for initializing Mondrian forest (seconds) = %f' % (time_init)
#    print 'Time for executing mondrianforest.py (seconds) = %f' % (time_method_sans_init)
#    print 'Total time for executing mondrianforest.py, including init (seconds) = %f' % (time_method)
#    print 'Time for prediction/evaluation (seconds) = %f' % (time_prediction)
#    print 'Total time (Loading data/ initializing / running / predictions / saving) (seconds) = %f\n' % (time_total)
#
#if __name__ == "__main__":
#    main()
