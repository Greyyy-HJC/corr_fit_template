# %%
import numpy as np
import gvar as gv
import matplotlib.pyplot as plt


fig_width = 6.75 # in inches, 2x as wide as APS column
gr        = 1.618034333 # golden ratio
fig_size  = (fig_width, fig_width / gr)
plt_axes = [0.1,0.12,0.85,0.8]
errorp = {"markersize": 5, "mfc": "none", "linestyle": "none"} # circle
errorb = {"markersize": 5, "mfc": "none", "linestyle": "none", "capsize": 3, "elinewidth": 1} # circle
errorl = {"markersize": 5, "mfc": "none", "capsize": 3, "elinewidth": 1} # circle with line
fs_p = {"fontsize": 13} # font size of text, label, ticks
ls_p = {"labelsize": 13}


def errorbar_plot(x, y, yerr, title, ylim=None):
    #* this is a general plot function, so put it here

    fig = plt.figure(figsize=fig_size)
    ax = plt.axes(plt_axes)
    ax.errorbar(x, y, yerr, **errorb)
    ax.tick_params(direction='in', **ls_p)
    ax.grid(linestyle=':')
    ax.set_ylim(ylim)
    plt.title(title, fs_p)
    # plt.legend()
    plt.savefig('fig/'+title+'.pdf', transparent=True)
    plt.show()


def pt2_to_meff(pt2_ls):
    meff_ls = []
    for i in range(len(pt2_ls)-1):
        val = np.log(pt2_ls[i]) - np.log(pt2_ls[i+1])
        meff_ls.append(val)
    return meff_ls

def pt2_pt3_to_R(tsep, tau_ls, pt2_0_ls, pt2_mom_ls, pt3_ls):
    #!# here pt2 list should be completed one started from tsep=0
    #!# tau list matches with pt3 list

    ratio1 = np.array( pt3_ls / pt2_0_ls[tsep] )

    #!# different from the formula 19 in the paper, coz we have zero momentum at sink
    ratio2 = []
    for tau in tau_ls:
        val1 = pt2_0_ls[tau] * pt2_0_ls[tsep] * pt2_mom_ls[tsep - tau]
        val2 = pt2_mom_ls[tau] * pt2_mom_ls[tsep] * pt2_0_ls[tsep - tau]

        ratio2.append( val1 / val2 )

    ratio2 = np.array(ratio2) 

    R_ls = ratio1 * (ratio2**0.5)

    return R_ls