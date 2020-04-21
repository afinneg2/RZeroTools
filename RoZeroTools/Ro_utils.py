import numpy as np 
import pandas as pd 
from scipy.stats import gamma, lognorm

def load_JHU_deaths(f_i = None, agg_by_state = True):
    """Fetch cumulative deaths from JHUs Covid github with optional aggregation by state. Note: this is not number of deaths occuring on each day
    
    Keyword Arguments:
        f_i {[type]} -- [description] (default: {None})
        agg_by_state {bool} -- [description] (default: {True})
    
    Returns:
        [type] -- [description]
    """
    
    if f_i is None:
        f_i = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv"
    df = pd.read_csv(f_i)
    #df = df.loc[df.iloc[:,5] != "Unassigned", : ].copy()
    cols = [6]
    cols.extend(np.arange(12, df.shape[1]))
    if agg_by_state:
        df = df.groupby("Province_State").apply(lambda x : x.iloc[:, cols[1:]].sum(axis = 0))
    
    return df

def load_state_age_demographic():

    #import csv from url: census.gov
    url = 'https://www2.census.gov/programs-surveys/popest/tables/2010-2018/state/asrh/sc-est2018-agesex-civ.csv'

    df = pd.read_csv(url)
    # delete male- and female-specific rows (Sex = 1 or 2)
    df = df[df.SEX != 1]
    df = df[df.SEX != 2]
    # delete total state population rows (Age = 999)
    df = df[df.AGE != 999]
    # delete national population rows (Name = United States)
    df = df[df.NAME != 'United States']
    #change specific ages to age-ranges

    df = df.replace({'AGE':[0,1,2,3,4,5,6,7,8,9]}, '0-9')
    df = df.replace({'AGE':[10,11,12,13,14,15,16,17,18,19]}, '10-19')
    df = df.replace({'AGE':[20,21,22,23,24,25,26,27,28,29]}, '20-29')
    df = df.replace({'AGE':[30,31,32,33,34,35,36,37,38,39]}, '30-39')
    df = df.replace({'AGE':[40,41,42,43,44,45,46,47,48,49]}, '40-49')
    df = df.replace({'AGE':[50,51,52,53,54,55,56,57,58,59]}, '50-59')
    df = df.replace({'AGE':[60,61,62,63,64,65,66,67,68,69]}, '60-69')
    df = df.replace({'AGE':[70,71,72,73,74,75,76,77,78,79]}, '70-79')
    df = df.replace({'AGE':[80,81,82,83,84,85]}, '>80')

    #new df with only pertinent columns
    df = df[['NAME', 'AGE', 'POPEST2018_CIV']]

    # sum same age-ranges by states
    state_ages_coarse = df.groupby(['NAME', 'AGE']).sum()
    state_ages_coarse = state_ages_coarse.reset_index()
    return state_ages_coarse


def make_infect_to_death_pmf( params_infect_to_symp = (1.62, 0.42),  params_symp_to_death = (18.8,  0.45), 
                             distrib_infect_to_symp = "lognorm" ,  distrib_symp_to_death = "gamma" ):
    """Construct numpy array representing pmf of distribution of days from infection to death. 

    Keyword Arguments:
        params_infect_to_symp {tuple} -- [description] (default: {(1.62, 0.42)})
        params_symp_to_death {tuple} -- [description] (default: {(18.8,  0.45)})
        distrib_infect_to_symp {str} -- [description] (default: {"lognorm"})
        distrib_symp_to_death {str} -- [description] (default: {"gamma"})

    Returns:
        [type] -- [description]
    """
    def make_gamma_pmf(mean, cv, truncate = 4):
        rv = gamma(a = 1./(cv)**2, scale = mean*(cv)**2)
        lower = 0 
        upper = int(np.ceil( mean + cv*mean*truncate) )
        cdf_samples = [rv.cdf(x) for x in range(lower, upper + 1, 1)]
        pmf = np.diff(cdf_samples)  
        pmf  = pmf / pmf.sum() ## normalize following truncation
        return pmf
        
    def make_lognorm_pmf( log_mean, log_sd, truncate=4):
        rv = lognorm( s = log_sd, scale = np.exp(log_mean),  )
        lower = 0 
        mean, var = rv.stats(moments = 'mv')
        upper = int(np.ceil( mean + np.sqrt(var)*truncate) )
        cdf_samples = [rv.cdf(x) for x in range(lower, upper + 1, 1)]
        pmf = np.diff(cdf_samples)  
        pmf  = pmf / pmf.sum() ## normalize following truncation
        return pmf
        
    if distrib_infect_to_symp.lower() == "lognorm":
        pmf_i_to_s = make_lognorm_pmf(*params_infect_to_symp)
    else:
        raise NotImplementedError()
    if  distrib_symp_to_death.lower() == "gamma":
        pmf_s_to_d = make_gamma_pmf(*params_symp_to_death)
    else:
        raise NotImplementedError()
    
    pmf_i_to_d =  np.convolve(pmf_i_to_s, pmf_s_to_d )
    ## truncate and renormalize
    cdf = np.cumsum(pmf_i_to_d)
    min_x = np.argmax( cdf >= 0.005  )  
    max_x = len(cdf) - np.argmax( (1. - cdf[::-1]) >= 0.005 ) 
    pmf_i_to_d[:min_x] = 0.
    pmf_i_to_d = pmf_i_to_d[: max_x]/ np.sum(pmf_i_to_d[: max_x])
    
    return pmf_i_to_d

def make_gamma_pmf(mean = 18.8, cv = 0.45, truncate = 3. ):
    """[summary]
    
    Keyword Arguments:
        mean {float} -- default is from lancet article (default: {18.8})
        cv {float} -- [oefficient of variation = std/mean efault is from lancet article  (default: {0.45})
        truncate {float} -- number of stds beyond the mean at which truncation occres
    
    Returns:
        [np.ndarrray] -- 1d array represting pmf
    """
    rv = gamma(a = 1./(cv)**2, scale = mean*(cv)**2)
    lower = 0 
    upper = int(np.ceil( mean + cv*mean*truncate) ) ## cv*mean = std
    cdf_samples = [rv.cdf(x) for x in range(lower, upper + 1, 1)]
    pmf = np.diff(cdf_samples)
    pmf  = pmf / pmf.sum() ## normalize following truncation
    return pmf
    