# %%
import numpy as np
import h5py as h5
import gvar as gv

#!# A3: np.imag( UU - DD )
#!# V4: np.real( UU - DD )

#* hadron: 'proton', 'proton_np'
#* flavor: 'UU', 'DD'
#* current: 'A3', 'V4'

p_sq_cut = 35  #* cut off of p^2

def find_key(dict, key_word):
    for key in dict:
        if key_word in key:
            return key

def find_data_2pt(hadron):
    fname = 'spec_4D_a12m130_a_tslice_avg_cfgs_300-5295_srcs_0-31_fft_n6.h5'

    basekey = 'gf1p0_w3p0_n30_M51p2_L520_a3p0/spec_4D/ml0p00195'
    basekey_2 = '4D_correlator/spin_avg'

    myfile = h5.File(fname, 'r')[basekey]
    data = myfile[hadron][basekey_2]

    return data


def find_data_3pt(flavor, current, tsep):
    file_name = 'formfac_4D_a12m130_a_proton_{}_{}_cfgs_300-5295_srcs_0-31_fft_n6.h5'

    basekey = 'gf1p0_w3p0_n30_M51p2_L520_a3p0/formfac_4D/ml0p00195' 
    basekey_2 = 'momentum_current'

    fname = file_name.format(flavor, current)
    myfile = h5.File(fname, 'r')[basekey]

    tsep_key = 'proton_{}_tsep_{}_sink_mom_px0_py0_pz0'.format(flavor, tsep)

    data = myfile[tsep_key][current][basekey_2]

    return data

def rotation_avg(data_ls):
    #!# need to separate correlators by their pz, not just the magnitude of the momentum
    #!# just take pz = 0 here, so it is (0,:,:)

    #* for (1000, xx, 13, 13, 13)

    hash_dic = {}

    for px in range(-6, 7):
        for py in range(-6, 7):
            pz = 0
            p_sq = px ** 2 + py ** 2 + pz ** 2
            if p_sq <= p_sq_cut: #* cut off
                hash_key = 'p_sq_{}_pz_{}'.format(p_sq, pz)
                if hash_key not in hash_dic:
                    hash_dic[hash_key] = []
                hash_dic[hash_key].append( data_ls[:,:,pz,py,px] )

    data_avg_ls = []
    hash_key_ls = []
    for key in hash_dic:
        hash_key_ls.append(key)
        data_avg_ls.append( np.average(hash_dic[key], axis=0) ) # shape = (len(hash), 1000, xx)

    data_avg_ls = np.swapaxes(data_avg_ls, 0, 1)
    data_avg_ls = np.swapaxes(data_avg_ls, 1, 2) # shape = (1000, xx, len(hash))
    
    return hash_key_ls, data_avg_ls

def main():
    data_set = {}

    #!# 2pt
    p_data = np.array( find_data_2pt('proton') )
    p_np_data = np.array( find_data_2pt('proton_np') )

    temp = ( p_data + p_np_data ) / 2

    #* take the real part of 2pt data
    temp = np.real(temp)

    hash_key_ls, temp_avg = rotation_avg(temp) #* rotation average
    data_set['2pt'] = temp_avg


    #!# 3pt
    for tsep in range(3, 12):
        tsep = str(tsep)

        #* A3
        current = 'A3'
        uu_data = np.array( find_data_3pt('UU', current, tsep) )
        dd_data = np.array( find_data_3pt('DD', current, tsep) )

        temp = np.imag( uu_data - dd_data )
        hash_key_ls, temp_avg = rotation_avg(temp) #* rotation average

        data_set['{}_tsep_{}'.format(current, tsep)] = temp_avg

        #* V4
        current = 'V4'
        uu_data = np.array( find_data_3pt('UU', current, tsep) )
        dd_data = np.array( find_data_3pt('DD', current, tsep) )

        temp = np.real( uu_data - dd_data )
        hash_key_ls, temp_avg = rotation_avg(temp) #* rotation average

        data_set['{}_tsep_{}'.format(current, tsep)] = temp_avg

    print([key for key in data_set])
    print([np.shape(data_set[key]) for key in data_set])

    data_set_avg = gv.dataset.avg_data(data_set)

    return hash_key_ls, data_set_avg


hash_key_ls, data_set_avg = main()

#* make the dict as [hash_key][2pt/3pt][:]
data_set_tidy = {}
for hash_key in hash_key_ls:
    data_set_tidy[hash_key] = gv.BufferDict()


for key in data_set_avg:
    for idx in range(len(hash_key_ls)):
        hash_key = hash_key_ls[idx]
        data_set_tidy[hash_key][key] = np.swapaxes(data_set_avg[key], 0, 1)[idx]


gv.dump(data_set_tidy, '../dump/data_set_tidy')


# %%
