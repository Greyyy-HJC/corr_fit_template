# %%
import numpy as np
import gvar as gv
import lsqfit as lsf
from fit_module import Fit
from prior import *
from funcs import *
from plot import *


data_set_tidy = gv.load('dump/data_set_tidy')

# print([key for key in data_set_tidy])

def data_check_meff(data_set_tidy, p_sq):
    hash_key = 'p_sq_{}_pz_0'.format(p_sq)
    pt2_ls = data_set_tidy[hash_key]['2pt']
    t_ls = np.arange(len(pt2_ls))
    meff_ls = pt2_to_meff(pt2_ls)

    errorbar_plot(t_ls[:-1], [v.mean for v in meff_ls], [v.sdev for v in meff_ls], 'meff_{}'.format(hash_key), ylim=[0, 1.5])

    meff = np.mean(meff_ls[6:12])


    # print('meff p_sq = {}'.format(p_sq))
    # print(meff)

    return meff
    

def do_fit(mom_ls, pt2_n, pt3_n):
    a12m130_fit = Fit(prior_ho_a12m130, pt2_n, pt3_n, include_2pt=True, include_3pt=True)

    pt2_t = {}
    pt2_t['p_sq_0_pz_0'] = np.arange(4, 12)
    for mom in mom_ls:
        pt2_t['p_sq_{}_pz_0'.format(mom)] = np.arange(4, 12)


    t_ls = []
    tau_ls = []
    tau_cut = 2
    for t in range(4, 10):
        for tau in range(tau_cut, t+1-tau_cut):
            t_ls.append(t)
            tau_ls.append(tau)

    pt3_A3 = {}
    pt3_A3['p_sq_0_pz_0'] = [t_ls, tau_ls]
    for mom in mom_ls:
        pt3_A3['p_sq_{}_pz_0'.format(mom)] = [t_ls, tau_ls]


    t_ls = []
    tau_ls = []
    tau_cut = 2
    for t in range(4, 10):
        for tau in range(tau_cut, t+1-tau_cut):
            t_ls.append(t)
            tau_ls.append(tau)

    pt3_V4 = {}
    pt3_V4['p_sq_0_pz_0'] = [t_ls, tau_ls]
    for mom in mom_ls:
        pt3_V4['p_sq_{}_pz_0'.format(mom)] = [t_ls, tau_ls]


    # temp = gv.load('dump/post')
    # p0 = {}
    # for key in temp:
    #     p0[key] = temp[key].mean

    fit_res, corr = a12m130_fit.fit(data_set_tidy, pt2_t, pt3_A3, pt3_V4, mom_ls, best_p0=None, corr=True)
    post = fit_res.p

    # print(fit_res.format(100))

    print('Q = {}'.format(fit_res.Q))

    print('A3 of mom 0')
    for i in range(pt3_n):
        print([post['A3_{}{}_0'.format(*[j,i])] for j in range(i+1)])

    for mom in mom_ls:
        print('A3 of mom {}'.format(mom))
        for i in range(pt3_n):
            print([post['A3_{}{}_{}'.format(*[j,i,mom])] for j in range(pt3_n)])

    print('\n')

    return fit_res, a12m130_fit.fit_func, corr


# %% 
#!# check meff plot of mom n
data_check_meff(data_set_tidy, 4)

#!# check ratio plot of mom n
fit_on_data_R(data_set_tidy, mom=2, current='V4', title='fit on data R mom 2', ylim=None)

# %%
#!# check disp relation
p_sq_ls = [0, 1, 2, 4, 5, 8, 9, 10, 13, 16, 17, 18, 20, 25, 32]
p_sq_GeV_ls = [v * 0.046 for v in p_sq_ls] #* p_sq * (2pi * 0.197 / L / a)**2

meff_ls = []
for p_sq in p_sq_ls:
    meff_ls.append( data_check_meff(data_set_tidy, p_sq) )

print(meff_ls)

meff_sq_ls = [(v**2) * 2.695 for v in meff_ls] #* meff * (0.197/0.12)**2

errorbar_plot(p_sq_GeV_ls, [v.mean for v in meff_sq_ls], [v.sdev for v in meff_sq_ls], 'meff_dispersion_GeV')



# %%
#!# check dispersion relation of all mom and fit to get k and b of disp relation

def fcn(x, p):
    return p['k'] * x + p['b']

priors = gv.BufferDict()
priors['k'] = gv.gvar(0, 10)
priors['b'] = gv.gvar(1, 10)

fit_result = lsf.nonlinear_fit(data=(np.array(p_sq_GeV_ls), meff_sq_ls), prior=priors, fcn=fcn, maxit=10000, fitter='scipy_least_squares')

print(fit_result) #* k ~ 1 means E^2 = p^2 + m^2 holds



# %%
#!# fit mom in []
mom_ls = [1] #!# included in the fit
pt2_n = 3
pt3_n = 3
fit_res, fcn_mom, corr = do_fit(mom_ls, pt2_n, pt3_n)
print([key for key in corr])


# %%
print(np.shape(corr['pt3_A3_0', 'pt3_A3_1']))
print(corr['pt3_A3_0', 'pt3_A3_1'])


# %%
#!# fit mom lists: [0], [0, 1] ...
p_sq_ls_non_zero = [1, 2, 4, 5, 8, 9, 10, 13, 16, 17, 18, 20, 25, 32]


pt2_n = 3
pt3_n = 3
A3_00_0_ls = []
for mom_ls in [p_sq_ls_non_zero[:i] for i in range(len(p_sq_ls_non_zero)+1)]:
    fit_res, fcn_mom, corr = do_fit(mom_ls, pt2_n, pt3_n)
    A3_00_0_ls.append( fit_res.p['A3_00_0'] )

gv.dump(A3_00_0_ls, 'dump/A3_00_0_ls_till_p_sq_{}_n{}'.format(*[p_sq_ls_non_zero[-1], pt3_n]))

errorbar_plot([0]+p_sq_ls_non_zero, [v.mean for v in A3_00_0_ls], [v.sdev for v in A3_00_0_ls], 'A3_00_0_different_mom_ls_till_p_sq_{}_n{}'.format(*[p_sq_ls_non_zero[-1], pt3_n]), ylim=[1.1, 1.5])



# %%
# fcn = fcn_mom(mom_ls)
# post = fit_res.p

# gv.dump(post, 'dump/post')



# %%
