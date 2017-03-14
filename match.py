from __future__ import print_function

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

# make sure match binaries are accessible
if 'match2.6' not in os.environ['PATH']:
    os.environ['PATH'] += ":/$HOME/match2.6/bin"

zc_names = ['age_low', 'age_high', 'dmod', 'sfr', 'sfr_p', 'sfr_m',
            'feh', 'feh_p', 'feh_m', 'dfeh', 'dfeh_p', 'dfeh_m',
            'massfrac', 'massfrac_p', 'massfrac_m']

# makefake 100000 15 28 -0.5 5 29 28 -snr=5
def makefake(out_dir, outfile, nstars=100000, mag_low=15, 
             mag_high=30, color_blue=-0.5, color_red=5, 
             completeness_1=29, completeness_2=28, **kwargs):
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

def write_par(infile, outfile, out_dir, dist, filter1, filter2='WFIRST_H158',
              zp1=29, zp2=28, age=9.7, feh=-1.75, age_spacing=0.1):
    with open(infile) as f:
        template = f.read()
    if 'fake' in infile:
        outstr = template.format(dist, filter1, filter2, zp1, zp2, age, age+age_spacing, feh)
    elif 'calcsfh' in infile:
        age_grid = make_age_grid(age, spacing=age_spacing)
        outstr = template.format(dist, filter1, filter2, zp1, zp2, age_grid)
    outpath = os.path.join(out_dir, outfile)
    with open(outpath, 'w') as f:
        f.write(outstr)

def make_age_grid(target_age, num_bins=15, spacing=0.1):
    age_range = num_bins * spacing
    age_low = target_age - age_range*(2/3)
    age_high = target_age + age_range/3 - spacing
    age_left = np.linspace(age_low, age_high, num_bins)
    age_right = age_left + spacing
    gridstr = '{}\n'.format(num_bins)
    gridstr += '\n'.join(['    {:.2f} {:.2f}'.format(age_left[i], age_right[i]) for i in range(num_bins)])
    return gridstr

# fake test4.fakepar fake.out -fake=fake.txt
def fake(out_dir, fakepar, outfile, infile, verbose=False, mist=True):
    parpath = os.path.join(out_dir, fakepar)
    inpath = os.path.join(out_dir, infile)
    outpath = os.path.join(out_dir, outfile)
    process = ['fake', parpath, outpath, '-fake='+inpath]
    if mist:
        process += ['-MIST_fast', '-gir16']
    ps = sp.Popen(process, stdout=sp.PIPE)
    output, err = ps.communicate()
    if verbose:
        print(output, err)

# calcsfh calcsfh.par makefake.out fake.out sfh.out -MIST_fast
def calcsfh(out_dir, parfile, fakefile, makefakefile, outfile, verbose=False, mist=True):
    #sfhpath = os.path.join(match_dir, 'bin/calcsfh')
    parpath = os.path.join(out_dir, parfile)
    fakepath = os.path.join(out_dir, fakefile)
    makefakepath = os.path.join(out_dir, makefakefile)
    outpath = os.path.join(out_dir, outfile)
    process = ['calcsfh', parpath, makefakepath, fakepath, outpath]
    if mist:
        process += ['-MIST_fast', '-gir16']
    ps = sp.Popen(process, universal_newlines=True, 
                  stdin=sp.PIPE, stdout=sp.PIPE)
    if verbose:
        while True:
            nextline = ps.stdout.readline()
            if nextline == '' and ps.poll() is not None:
                break
            elif nextline.startswith('Av='):
                ps.stdin.write('\n')
            sys.stdout.write(nextline)
            sys.stdout.flush()
    ps.communicate()

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

def add_systematic(out_dir, systematic, filt):
    sys_dir = '{:.3f}mag_{}'.format(systematic,filt)
    sys_path = os.path.join(out_dir, sys_dir)
    os.makedirs(sys_path, exist_ok=True)
    fake = np.loadtxt(os.path.join(out_dir,'fake.out'))
    fake_new = fake.copy()
    if filt == 'H158':
        fake_new[:,1] += systematic
    else:
        fake_new[:,0] += systematic
    np.savetxt(os.path.join(sys_path,'fake.out'), fake_new, fmt='%.3f', delimiter=' ')
    return sys_dir

def parse_zc_file(zc_file):
    # WFIRST_X625/10Mpc/8.00yr/-2.00dex/zcombine.out
    # WFIRST_X625/10Mpc/8.00yr/-2.00dex/0.002mag_H158/zcombine.out
    zc_file_split = zc_file.split('/')
    filt = zc_file_split[0].split('_')[1]
    dist = float(zc_file_split[1].split('Mpc')[0])
    age = float(zc_file_split[2].split('yr')[0])
    feh = float(zc_file_split[3].split('dex')[0])
    if zc_file_split[4].find('mag_') == -1:
        systematic = 'None'
    else:
        systematic = zc_file_split[4].replace('mag_',' ')
    return filt, dist, age, feh, systematic

def make_row(zc_file, zc_names=zc_names):
    filt, dist, age, feh, systematic = parse_zc_file(zc_file)
    df_orig = pd.read_csv(zc_file, delim_whitespace=True, skiprows=6, names=zc_names)
    df = df_orig[(df_orig.age_low == age)]
    if float(age) > 9.75:
        age_spacing = 0.08
    else:
        age_spacing = 0.15
    mass_before = df_orig[(df_orig.age_low > age) & (df_orig.age_low < age+age_spacing)].massfrac.sum()
    df = df.assign(mass_thisbin=df.massfrac-mass_before)
    df = df.assign(filt=filt)
    df = df.assign(dist=dist)
    df = df.assign(age=age)
    df = df.assign(feh_input=feh)
    df = df.assign(systematic=systematic)
    return df

def make_df(zc_list):
    df = make_row(zc_list[0])
    for zc_file in zc_list[1:]:
        df = pd.concat([df, make_row(zc_file)], ignore_index=True)
    return df

def merge_csvs(dflist):
    name = dflist[0].split('_')[1]
    df = pd.read_csv(dflist[0])
    for i in dflist[1:]:
        df = pd.concat([df, pd.read_csv(i)], ignore_index=True)
    gr = df.groupby(['systematic','dist','age','feh_input'], group_keys=False, as_index=False)
    s = gr[['mass_thisbin','feh','feh_p','feh_m']].agg([np.mean, np.median, np.std])
    s.to_csv('{}_grid.csv'.format(name))

def check_clobber_condition(out_dir, clobber):
    if clobber:
        clobber_condition = False
    else:
        clobber_condition = os.path.exists(os.path.join(out_dir,'zcombine.out'))
    return clobber_condition

def run(inlist, clobber=False, systematics=False, makenewfake=False):
    dmod, filter1, age, feh = inlist
    if (dmod < 28.5):
        dist = '4Mpc'
    elif (dmod > 28.5) & (dmod < 29):
        dist = '6Mpc'
    elif (dmod > 29) & (dmod < 29.7):
        dist = '8Mpc'
    elif (dmod > 29.7):
        dist = '10Mpc'
    runstr = '{} at {}, age {:.2f}, [Fe/H] {:.2f}'.format(filter1, dist, age, feh)
    if float(age) > 9.75:
        age_spacing = 0.05
    else:
        age_spacing = 0.1
    out_dir = os.path.join(filter1, dist, '{:.2f}yr'.format(age), '{:.2f}dex'.format(feh))
    os.makedirs(out_dir, exist_ok=True)
    clobber_condition = check_clobber_condition(out_dir, clobber)
    if not clobber_condition:
        print('  Running ' + runstr)
        if makenewfake:
            makefake(out_dir, 'makefake.out', snr=5)
        else:
            makefake_origpath = os.path.join(os.getcwd(), 'makefake.out')
            makefake_newpath = os.path.join(os.getcwd(), out_dir, 'makefake.out')
            shutil.copy(makefake_origpath, makefake_newpath)
        write_par('fake_template.fakepar', 'fake.fakepar', out_dir, dmod, filter1,
                  age=age, feh=feh, age_spacing=age_spacing)
        write_par('calcsfh_template.par', 'calcsfh.par', out_dir, dmod, filter1,
                  age=age, feh=feh, age_spacing=age_spacing)
        fake(out_dir, 'fake.fakepar', 'fake.out', 'makefake.out')
        calcsfh(out_dir, 'calcsfh.par', 'makefake.out', 'fake.out', 'sfh.out')
        zcombine(out_dir, 'sfh.out', 'zcombine.out')
    else:
        print('  Already ran ' + runstr)
    if systematics:
        for systematic in np.linspace(0.002,0.008,4):
            for filt in [filter1.split('_')[-1],'H158']:
                sys_dir = add_systematic(out_dir, systematic, filt)
                clobber_condition_sys = check_clobber_condition(sys_dir, clobber)
                if not clobber_condition_sys:
                    print('    Running systematics at dmod {} for {} with {} mag'.format(dmod, filt,systematic))
                    new_fake_out = os.path.join(sys_dir, 'fake.out')
                    new_sfh_out = os.path.join(sys_dir, 'sfh.out')
                    new_zcombine_out = os.path.join(sys_dir, 'zcombine.out')
                    calcsfh(out_dir, 'calcsfh.par', 'makefake.out', new_fake_out, new_sfh_out)
                    zcombine(out_dir, new_sfh_out, new_zcombine_out)

if __name__ == '__main__':
    dmod_cycle = cycler(dmod=[28.0105, 30]) # [28.0105, 28.8906, 29.5155, 30]
    filt_cycle = cycler(filt=['WFIRST_X606', 'WFIRST_X625'])
    age_cycle = cycler(ages=[8., 9., 10.])
    feh_cycle = cycler(fehs=[-2., -1., 0.])
    param_cycle = (dmod_cycle * filt_cycle * feh_cycle * age_cycle).by_key()
    inlist = list(zip(*[param_cycle['dmod'], param_cycle['filt'], param_cycle['ages'], param_cycle['fehs']]))
    if not os.path.isfile(os.path.join(os.get_cwd(), 'makefake.out')):
        makefake(os.getcwd(), 'makefake.out', snr=5)
    for i in range(1):
        print('Beginning run {}'.format(i+1))
        #for filt in ['WFIRST_X606', 'WFIRST_X625', 'WFIRST_X670', 'WFIRST_Z087']:
        p = mp.Pool(int(mp.cpu_count()/2))
        p.map(run, inlist)
        p.close()
        p.join()
        for filt in filt_cycle.by_key()['filt']:
            zc_list = glob.glob('{}/*Mpc/*yr/*dex/zcombine.out'.format(filt)) + glob.glob('{}/*Mpc/*yr/*dex/*mag*/zcombine.out'.format(filt))
            df = make_df(zc_list)
            df.sort_values(by=['dist','age','feh_input'], inplace=True)
            os.makedirs(os.path.join(os.getcwd(), '{}_csvs'.format(filt)), exist_ok=True)
            df_path = os.path.join(os.getcwd(), '{0}_csvs/{0}_{1}_grid.csv'.format(filt,i))
            df.to_csv(df_path,index=False)
            print('DF written for run {} of filter {}'.format(i+1,filt))
    for filt in filt_cycle.by_key()['filt']:
        dflist = glob.glob('{}_*.csv'.format(filt))
        merge_csvs(dflist)
