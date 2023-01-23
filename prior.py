import numpy as np 
import gvar as gv 

mom_list = [1] # no zero-mom

def prior_ho_a09m310(pt2_n, pt3_n):
    prior = gv.BufferDict()

    #!# zero mom
    for i in range(pt3_n):
        for j in range(pt3_n):
            mi = np.minimum(j, i)
            ma = np.maximum(j, i)
            prior['A3_{}{}_0'.format(*[mi, ma])] = gv.gvar(0, 3)
            prior['V4_{}{}_0'.format(*[mi, ma])] = gv.gvar(0, 3)
        prior['z{}_0'.format(i)] = gv.gvar(0, 0.01)




    prior['E0_0'] = gv.gvar(0.6, 0.06)
    prior['log(dE1_0)'] = gv.gvar(-1.07, 0.3) # ln(0.94-0.6)
    prior['log(dE2_0)'] = gv.gvar(-0.69, 0.5) # ln(1.44-0.94)
    #prior['log(dE3)']

    prior['log(dEmax_pt2_0)'] = gv.gvar(-1.25, 0.5*5)
    prior['log(dEmax_pt3_0)'] = gv.gvar(-1.25, 0.5*5)




    #!# non zero mom
    for mom in mom_list:
        mo = '_'+str(mom)

        for i in range(pt3_n):
            for j in range(pt3_n):
                prior['A3_{}{}'.format(*[i, j])+mo] = gv.gvar(0, 3)
                prior['V4_{}{}'.format(*[i, j])+mo] = gv.gvar(0, 3)
            prior['z{}'.format(i)+mo] = gv.gvar(0, 0.008)





        Emean = np.sqrt(0.6 ** 2 + (2*np.pi*np.sqrt(mom)/48) ** 2)
        prior['E0'+mo] = gv.gvar(Emean, Emean/5)

        dE1mean =  np.sqrt((0.94) ** 2 + (2*np.pi*np.sqrt(mom)/48) ** 2) - Emean
        prior['log(dE1'+mo+')'] = gv.gvar(np.log(dE1mean), 0.5)

        dE2mean =  np.sqrt((1.44) ** 2 + (2*np.pi*np.sqrt(mom)/48) ** 2) - Emean - dE1mean
        prior['log(dE2'+mo+')'] = gv.gvar(np.log(dE2mean), 0.8)

        prior['log(dEmax_pt2'+mo+')'] = gv.gvar(-1.25, 0.5*10)
        prior['log(dEmax_pt3'+mo+')'] = gv.gvar(-1.25, 0.5*10)


    return prior