#from __future__ import print_function
import glob
import numpy as np
import os
import pandas as pd
import shutil
import subprocess as sp
import sys
from cycler import cycler
from functools import partial
import multiprocessing as mp
import argparse
import xarray as xr

# make sure match binaries are accessible
if 'match2.7' not in os.environ['PATH']:
    os.environ['PATH'] += ":/astro/store/phat2/projects/src/match2.7/bin"

zp1 = 28.7
zp2 = 29.2
age_spacing = 0.05

# makefake 100000 20 30 -2 5 29 28 -snr=5
def makefake(out_dir, outfile, nstars=100000, mag_low=15, 
             mag_high=30, color_blue=-2.0, color_red=5.0, 
             completeness_1=zp1, completeness_2=zp2, **kwargs):
    '''Wrapper around makefake
    
    Params:

        out_dir : path
            Directory to write makefake output to
        outfile : str
            Name of makefake output file
        nstars : int
            Number of fake stars to make
        mag_low : scalar
            Minimum (brightest) magnitude for fake stars
        mag_high : scalar
            Maximum (dimmest) magnitude for fake stars
        color_blue : scalar
            Bluest allowed color for fake stars
        color_red : scalar
            Reddest allowed color for fake stars
        completeness_1 : scalar
            50% completeness limit for blue filter
        completeness_2 : scalar
            50% completeness limit for red filter
        **kwargs : optional
            Extra flags for makefake (see match readme)

    Outputs: fake artificial star test file
    '''
    params = list(np.array([nstars, mag_low, mag_high, color_blue, color_red, 
                  completeness_1, completeness_2]).astype(str))
    flags = ['-{}={}'.format(k, v) for k, v in kwargs.items()]
    params += flags
    outpath = os.path.join(out_dir, outfile)
    with open(outpath, 'w') as out:
        ps = sp.Popen(['makefake', *params], 
               universal_newlines=True, stdout=out)
        ps.communicate()
    print('{} fake stars made'.format(nstars))

def make_age_grid(target_age, num_bins=24, max_age=10.1, age_spacing=age_spacing):
    target_age = float(target_age)
    age_range = num_bins * age_spacing
    age_low = target_age - age_range*(3/4)
    age_high = target_age + age_range/4 - age_spacing
    if age_high > max_age:
        num_bins -= int(np.round((age_high - max_age)/age_spacing,decimals=0))
        age_high = max_age
    age_left = np.linspace(age_low, age_high, num_bins)
    age_right = age_left + age_spacing
    gridstr = '{}\n'.format(num_bins)
    gridstr += '\n'.join(['    {:.2f} {:.2f}'.format(age_left[i], age_right[i]) for i in range(num_bins)])
    return gridstr

def write_par(infile, outfile, out_dir, dmod, filter1, age, feh=None, sfr=None, filter2='WFIRST_H158',
              zp1=zp1, zp2=zp2, age_spacing=age_spacing):
    with open(infile) as f:
        template = f.read()
    if 'fake' in infile:
        outstr = template.format(dmod, filter1, filter2, zp1, zp2, float(age),
            float(age)+age_spacing, sfr, float(feh))
    elif 'calcsfh' in infile:
        age_grid = make_age_grid(age, age_spacing=age_spacing)
        outstr = template.format(dmod, filter1, filter2, zp1, zp2, age_grid)
    outpath = os.path.join(out_dir, outfile)
    with open(outpath, 'w') as f:
        f.write(outstr)

# fake fake.par fake.out -fake=makefake.out
def fake(out_dir, fakepar, outfile, infile, model, verbose=False):
    parpath = os.path.join(out_dir, fakepar)
    inpath = os.path.join(out_dir, infile)
    outpath = os.path.join(out_dir, outfile)
    process = ['fake', parpath, outpath, '-fake='+inpath]
    if model != 'Padua2006_CO_AGB':
        process += ['-{}'.format(model), '-gir16']
    ps = sp.Popen(process, stdout=sp.PIPE)
    output, err = ps.communicate()
    if verbose:
        print(output, err)

# calcsfh calcsfh.par makefake.out fake.out sfh.out -MIST_fast -gir16 -verb
def calcsfh(out_dir, parfile, fakefile, makefakefile, outfile, model, verbose=False):
    #sfhpath = os.path.join(match_dir, 'bin/calcsfh')
    parpath = os.path.join(out_dir, parfile)
    fakepath = os.path.join(out_dir, fakefile)
    makefakepath = os.path.join(out_dir, makefakefile)
    outpath = os.path.join(out_dir, outfile)
    infopath = os.path.join(out_dir, outfile.replace('.out','_info.out'))
    process = ['calcsfh', parpath, makefakepath, fakepath, outpath, '-verb']
    if model != 'Padua2006_CO_AGB':
        process += ['-{}'.format(model), '-gir16']
    with open(infopath, 'w') as out:
        ps = sp.Popen(process, 
               universal_newlines=True, stdout=out)
        output, err = ps.communicate()
        if verbose:
            print(output, err)

# zcombine sfh.out
def zcombine(out_dir, sfhfile, outfile, verbose=False):
    sfhpath = os.path.join(out_dir, sfhfile)
    outpath = os.path.join(out_dir, outfile)
    with open(outpath, 'w') as out:
        ps = sp.Popen(['zcombine', sfhpath], 
               universal_newlines=True, stdout=out)
        ps.communicate()
    if verbose:
        print('zcombine output written to {}'.format(outpath))

def read_sfh_info(sfhinfofile):
    with open(sfhinfofile) as f:
        info = f.read()
    nstars = int(info.split(' real stars read')[0].split('\n')[-1])
    fit = float(info.split(', fit=')[-1].split('\n')[0])
    info_dict = {'nstars':nstars,'fit':fit}
    return info_dict

def read_zc(zcfile):
    df = pd.read_csv(zcfile, delim_whitespace=True, usecols=[0,6,12], skiprows=6,
                     names=['age_bin', 'feh_meas', 'massfrac'], skipinitialspace=True, 
                     dtype={'age_bin': str, 'feh_meas': np.float64, 'massfrac': np.float64})
    df.set_index('age_bin', inplace=True)
    # df = df.assign(massdiff=df.massfrac.diff(periods=-1))
    # df.massdiff.iloc[-1] = 1 - (df.massdiff.sum() + (1 - df.massfrac.iloc[0]))
    # df = df.assign(weighted_feh=df.massdiff*df.feh)
    # row = df.loc[age]
    # mass_after = df[df.index.astype(float)<(float(age)-0.01)].massdiff.sum()
    # mass_before = df[df.index.astype(float)>(float(age)+0.01)].massdiff.sum()
    # zc_dict = {'massfrac':row.massdiff, 'feh_agebin':row.feh, 'feh_mean':df.weighted_feh.sum(),
    #     'mass_before':mass_before, 'mass_after':mass_after}
    return df

def run_core(out_dir, dmod, filter1, age, feh, sfr, model, verbose):
    shutil.copyfile(os.path.join(os.getcwd(),'makefake.out'),
        os.path.join(out_dir,'makefake.out'))
    write_par('fake_template.par', 'fake.fakepar', out_dir, dmod, filter1, age,
        feh=feh, sfr=sfr)
    write_par('calcsfh_template.par', 'calcsfh.par', out_dir, dmod, filter1, age)
    fake(out_dir, 'fake.fakepar', 'fake.out', 'makefake.out', model, verbose=verbose)
    calcsfh(out_dir, 'calcsfh.par', 'makefake.out', 'fake.out', 'sfh.out', model, verbose=verbose)
    zcombine(out_dir, 'sfh.out', 'zcombine.out', verbose=verbose)
    os.remove(os.path.join(out_dir,'makefake.out'))
    df = read_zc(os.path.join(out_dir, 'zcombine.out'))
    #info_dict = read_sfh_info(os.path.join(out_dir,'sfh_info.out'))
    return df

def run_test(out_dir, age):
    shutil.copyfile(os.path.join(os.getcwd(),'sfh_info_test.out'),
        os.path.join(out_dir,'sfh_info_test.out'))
    shutil.copyfile(os.path.join(os.getcwd(),'zcombine_test.out'),
        os.path.join(out_dir,'zcombine_test.out'))
    df = read_zc(os.path.join(out_dir, 'zcombine_test.out'))
    #info_dict = read_sfh_info(os.path.join(out_dir,'sfh_info_test.out'))
    return df

def run(inlist, r, model, age_spacing=age_spacing, verbose=False, test=False):
    filter1, dist, mass, age, feh = inlist
    dmod = 5*np.log10(dist*1e6)-5
    sfr = (10**mass) / (10**(float(age)+age_spacing) - 10**float(age))
    runstr = '    Run {}: {} at {} Mpc, {} logsolMass, age {}, [Fe/H] {}'.format(r, filter1, dist, mass, age, feh)
    print(runstr)
    out_dir = os.path.join(os.getcwd(), filter1, 'dist{}'.format(dist),
        'logSolMass{}'.format(mass), 'logYr{:.1f}'.format(float(age)).replace('.','p'),
        'dex{:.1f}'.format(float(feh)).replace('.','p').replace('-','_'))
    # print(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if test:
        df = run_test(out_dir, age)
    else:
        df = run_core(out_dir, dmod, filter1, age, feh, sfr, model, verbose)
    # d.loc[filter1, dist, mass, age, feh, str(r), df.columns, df.index] = df
    # values_dict = info_dict.copy()
    # values_dict.update(zc_dict)
    # for k,v in values_dict.items():
    #     d.loc[filter1, dist, mass, age, feh, str(r), k] = v
    #     print('    {} {} = {}'.format(runstr, k, v))
    # if test:
    #     os.remove(os.path.join(out_dir,'sfh_info_test.out'))
    #     os.remove(os.path.join(out_dir,'zcombine_test.out'))
    return filter1, dist, mass, age, feh, df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Padua2006_CO_AGB', help='Specify model',
        choices=['Padua2006_CO_AGB', 'MIST', 'MIST_fast', 'PARSEC'])
    parser.add_argument('--nproc', default=int(mp.cpu_count()/2), type=int,
        help='Number of processes; defaults to number of CPUs')
    parser.add_argument('--runs', default=20, type=int, help='Number of runs; defaults to 20')
    parser.add_argument('--verbose', action='store_true', help='Print all process output')
    # parser.add_argument('--append', action='store_true', help='Append to existing HDF file')
    parser.add_argument('--test', action='store_true', help='Run fake test just to see if it writes the file')
    args = parser.parse_args()

    print("Number of threads: {}".format(args.nproc))
    print("Using model {}".format(args.model))

    filt=['WFIRST_X606']#, 'WFIRST_X625', 'WFIRST_Z087'] # 'WFIRST_X606', , 'WFIRST_Z087'
    dist=[4]#, 6, 8, 10]
    mass=[7] # 6, 8
    #age=['{:.2f}'.format(a) for a in [8.5, 9.0, 9.5, 9.8, 10.0, 10.1]]
    age=['{:.2f}'.format(a) for a in [9.3]]#, 9.5, 9.7, 9.9, 10.0, 10.1]]
    feh=['{:.2f}'.format(f) for f in [-2.2]]#, -1.8, -1.3, -0.8, -0.5, -0.2, 0.0, 0.1]]
    filt_cycle = cycler(filt=filt)
    dist_cycle = cycler(dist=dist)
    mass_cycle = cycler(mass=mass)
    age_cycle = cycler(age=age)
    feh_cycle = cycler(feh=feh)
    vals = ['feh_meas', 'massfrac']#'mass_before', 'mass_after', 'feh_agebin', 'feh_mean', 'nstars', 'fit']
    age_bin = ['{:.2f}'.format(a) for a in np.arange(7.5, 10.2, 0.05)]
    runs = np.arange(1, args.runs+1) # + ['mean', 'median', 'std']
    param_cycle = (filt_cycle * dist_cycle * mass_cycle * feh_cycle * age_cycle).by_key()
    keylist = ['filt','dist','mass','age','feh','runs','age_bin','vals']
    dimlist = [filt, dist, mass, age, feh, runs, age_bin, vals]
    coord_dict = {keylist[i]:dimlist[i] for i in range(len(keylist))}
    inlist = list(zip(*[param_cycle[k] for k in keylist[:-3]]))
    d = xr.DataArray( np.zeros( [len(coord_dict[k]) for k in keylist] ), dims=keylist,
        coords=coord_dict)
    d.loc[:,:,:,:,:,:,:,:] = np.nan
    dpath = '{}_{}.nc'.format(args.model, '_'.join(filt))
    d.to_netcdf(dpath, mode='w')
    makefake(os.getcwd(), 'makefake.out', snr=5)
    for r in range(1, args.runs+1):
        print('Beginning run {} out of {}'.format(r, args.runs))
        p = mp.Pool(args.nproc)
        func = partial(run, r=r, model=args.model, verbose=args.verbose, test=args.test)
        try:
            output = p.map(func, inlist)
            p.close()
            p.join()
            for line in output:
                filter1, dist, mass, age, feh, df = line
                # print(filter1, dist, mass, age, feh)
                # print(df)
                d.loc[filter1, dist, mass, age, feh, r, df.index[0]:df.index[-1], :] = df.values
                # print(d.loc[filter1, dist, mass, age, feh, r, df.index[0]:df.index[-1], :])
                # for k,v in values_dict.items():
                #     d.loc[filter1, dist, mass, age, feh, str(r), k] = v
            d.to_netcdf(dpath, mode='w')
        except:
            print('Run {} failed!'.format(r))
            for i in sys.exc_info():
                print(i)
            d.to_netcdf(dpath, mode='w')
    #d.to_netcdf(dpath, mode='w')
    print('Done!')
