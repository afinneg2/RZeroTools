import numpy as np 
import pandas as pd 
from collections import OrderedDict
from scipy.stats import gamma, lognorm
import textwrap
from datetime import timedelta

import matplotlib.pyplot as plt
import seaborn as sns

######################################################
### LOAD DATA
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


### PROBABILITY DISTRIBUTIONS

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
    
### PLOTTING
def plot_shaded(data, figsize = (12,5), **kwargs ):
    """
    data - df -- index is x values, 
                 columns is [mean, upper ,lower]. If columns is multilevel level 0 stores names of different trances
                        level 1 stores mean, upper, lower
                 data is y values
    """
    
    fig, ax = plt.subplots(figsize = figsize)
    
    if isinstance(data.columns, pd.MultiIndex):
        for l in data.columns.levels[0]:
            data.loc[: , (l, "mean")].plot(ax = ax, label = l, marker= 'o')
            x_data = ax.get_lines()[-1].get_xdata()
            ax.fill_between(x_data,  
                            data.loc[: , (l, "lower")].values ,
                            data.loc[: , (l, "upper")].values ,
                           **kwargs)
    ax.legend()
    return fig

class Policy_Stats(object):
    """Class for relating changes in policy to changes in statictics of samples from the posterior
    distribution that are calcuatled over time intervals before and after the change. 
    """
    def __init__(self, policy_dates, max_before =7, max_after = 7):
        """
        policies — DataFrame - index : policy names
                               columns : states/provinces
        """
        self.max_before = max_before
        self.max_after = max_after
        self._policy_dates = Policy_Stats._preprocess_policy_dates(policy_dates)
        
        self._compare_intervals = self._policy_dates.groupby(level = 0).apply(
                                                                lambda x: Policy_Stats._make_compare_intervals(x[x.name]) 
                                                                        )
        self.change_samples = None
        self.change_interval = None
    @property
    def policy_dates(self):
        """
        pd.Series Multiindex with level0 — state and level1 — policy name 
                  Values are dates
        """
        return self._policy_dates.copy()
    @property
    def compare_intervals(self):
        """
        pd.DataFrame: index - level 0 - state; level 1 - date
                      columns before_start, before_end, after_start, after_end, policy
                      values - TimeStamps or string
        """
        return self._compare_intervals.copy()
    
    @staticmethod  
    def _preprocess_policy_dates(policy_dates):
        processed = policy_dates.applymap(lambda x: pd.Timestamp(x)
                                         ).dropna(axis =1, how = "all")
        processed = processed.transpose().stack()
        return  processed 
    
    @staticmethod  
    def _make_compare_intervals( p_series, max_before =7, max_after = 7, truncate_before = False ):
        """
        Construct time-intervals before and after implementation of a policy, considering the dates at
        which other policies are implemented.
        Inputs
        ------
            p_series — pd.Series: index - policies,
                                 values - dates implemented
            max_before - int: the number of days before implementation of policy to include
            max_after - int: the number of days after implementation policy (including the day implementation) to include
            truncate_before — If the "before" interval overlaps the implementation of a previous policy do we trucate 
                            to only the later part of the interval? 
        Returns
        -------
            compare_intervals- index - level 0 - state; level 1 - date
                                  columns before_start, before_end, after_start, after_end, policy
                                values - TimeStamps or string
        """
        
        p_series = p_series.squeeze().sort_values().dropna()
        dates_unique = [pd.Timestamp(x) for x in sorted(p_series.unique())]

        date_to_p = OrderedDict([])
        for i, d in enumerate(dates_unique):
            if not date_to_p:  ## the first date
                before_start = d + timedelta(days=-max_before) 
                before_end = d 
                after_start = d
                if i + 1 < len(dates_unique): ## there is another entry
                    after_end = min( dates_unique[i+1], d + timedelta(days=max_after) )
                else:
                    after_end = d + timedelta(days=max_after) 
            else:
                if truncate_before:
                    before_start = max( d+timedelta(days=-max_before), dates_unique[i-1] + timedelta(days=1) )
                else:
                    before_start = d + timedelta(days=-max_before)
                before_end = d 
                after_start = d
                if i + 1 < len(dates_unique): ## there is another entry
                    after_end = min( dates_unique[i+1], d + timedelta(days=max_after) )
                else:
                    after_end = d + timedelta(days=max_after) 
            date_to_p[d] = [ before_start, before_end, after_start, after_end ] 

        compare_intervals = pd.DataFrame.from_dict( date_to_p,
                                                   orient = "index", 
                                                   columns = [ "before_start", "before_end", "after_start", "after_end"])
        ## Add policies for each date
        compare_intervals["policy"] = ""
        for p, d in p_series.iteritems():
            if compare_intervals.loc[d, "policy"]:
                compare_intervals.loc[d, "policy"] = compare_intervals.loc[d, "policy"] + "_AND_" + p
            else:
                compare_intervals.loc[d, "policy"] = p            
        return compare_intervals
    
    def est_pct_change(self, samples_dict, policies):
        """
        Inputs
        ------
            samples_dict — dictionary: keys -states
                                       values - dataframe with dates as index and samples as columns
            policies -  None or list of strings appearing in self.policy_dates index, level1
        Returns:
            pct_change_interval — pd.DataFrame: index: level0 state; level1 policy,
                                           columns: mean, lower, upper
                                           values: percent change
            pct_samples — pd.DataFrame: index: level0 state; level1 policy,
                                           columns: samples index
                                           values: percent change
        """
        c_intervals = self.compare_intervals.copy()
        change_func =  lambda before, after: (after.mean(axis = 0) - before.mean(axis = 0)).divide(before.mean(axis = 0))*100.
        self.change_samples  = self._est_change(samples_dict, c_intervals, change_func = change_func,  policies = policies)
        ## Get credible interval
        self.change_interval= self._samples_to_cred_interval( self.change_samples , sample_axis = 1)
        return self.change_interval.copy(), self.change_samples.copy()

    @staticmethod
    def _est_change(samples_dict, c_intervals, change_func,  policies = None):
        """
        
        Inputs
        ------
            samples_dict — dictionary: keys -states
                                       values - dataframe with dates as index and samples as columns
            c_intervals - same as self.compare_intervals
            change_func - callable: arguments: before (pd.DataFrame), after (pd.DataFrame)
                                            where each columns is a sample and index is dates from before or after interval
                                    returns: series where index is samples and values are the change statisitic for the sample  
            policies -  None or list of strings appearing in self.policy_dates index, level1
        Returns
        -------
            change_samples — pd.DataFrame: index: level0 state; level1 policy,
                                           columns: samples index
                                           values: change statisitc
        """
        samples_dict= {s: df for s, df in samples_dict.items() if s in c_intervals.index.levels[0]} ## filter to states with policy data
        results_dict = {}
        for state, samples in samples_dict.items():
            ## Get dates and names of policies
            if policies is None:
                dates_interest =  list( c_intervals.loc[state,:].index )
            else:
                dates_interest = [d for d, row in c_intervals.loc[state,: ].iterrows() 
                                      if set(policies).intersection(row["policy"].split("_AND_"))  ]
            policies_interest =  list(c_intervals.loc[ [(state, d) for d in dates_interest] , "policy"])
            changes_by_sample = pd.DataFrame( index =  policies_interest, columns = samples.columns, data = 0. )
            ### Compute change at each policy/date
            for d, p in zip(dates_interest, policies_interest):
                before_start, before_end, after_start, after_end = c_intervals.loc[(state, d), 
                                                                         ["before_start", "before_end", "after_start", "after_end"]]
                changes_by_sample.loc[p, :] = change_func( samples.loc[ before_start:before_end , :],  
                                                    samples.loc[ after_start: after_end , :]).values
            ## Update with results for state
            results_dict[state] = changes_by_sample
        ## Combine to single dataframe
        keys_tmp = list(results_dict.keys())
        change_samples = pd.concat( [results_dict[k] for k in keys_tmp] , axis = 0,  keys = keys_tmp)
        ## Add policy_date to index
        change_samples["policy_date"] = pd.NaT
        for (state,date), row in c_intervals.iterrows():
             change_samples.loc[ (state, row["policy"]), "policy_date"] = date
        change_samples = change_samples.set_index( "policy_date" , append = True)
        change_samples.index.names = ["state/province" , "policy" , "policy_date"]
        return change_samples 

    def boxplot_changes_by_state(self, states = None, policies = None, change_samples =None ,
                                aspect= 3, height = 6, hspace = 1.):
        ####### Process inputs
        if change_samples is None:
            if self.change_samples is None:
                raise Exception("change_samples not set eiter provide as method argument or run est_pct_change")
            else:
                change_samples = self.change_samples
        ## states
        if states is None:
            states = list(change_samples.index.levels[0])
        else:
            assert all([s in change_samples.index.levels[0] for s in states]),"No data for some states in provided list" 
            change_samples = change_samples.loc[states, : ].copy()
        if policies is not None:
            mask = [ len(set(i[1].split("_AND_")).intersection(policies)) > 0 for i in change_samples.index ]
            change_samples_mask = change_samples_mask.loc[mask, :].copy()
        ####### Helper functions
        def format_xlab(s, width = 12):
            s = "\nAND\n".join([textwrap.fill(x, width = width) for x in s.split("_AND_")])
            return s
        def plot_func(data, y, color):
            policies_and_dates = data[["policy", "policy_date"]].drop_duplicates().sort_values(by = "policy_date").reset_index(drop =True)
            ax = sns.boxplot(data = data, x = "policy" , y = y, order = list(policies_and_dates["policy"].values),  whis=[5, 95])
            ax.set_xticklabels([format_xlab(t, width = 20) for t in policies_and_dates["policy"].values] , rotation = 90)
            return None
        ####### PLOT
        change_samples = change_samples.transpose().melt()
        g = sns.FacetGrid(change_samples, row="state/province", aspect= aspect, height = height,sharex=False, )
        g.map_dataframe(plot_func, y = "value")
        for ax in np.ravel(g.axes):
            ax.set_ylabel("Percent change in " + r'$R_e$')
            ax.set_ylim(None, 50)
            ax.grid(True)
        g.fig.subplots_adjust(hspace=hspace)
        return g.fig
    
    def boxplot_changes_by_policy(self, states = None, policies = None,  change_samples = None,
                                  y_label = "Percent change in " + r'$R_e$', aspect = 4, height = 6,y_lim = (-100, 100),
                                 hspace = 3.2 ):
        ####### Process inputs
        if change_samples is None:
            if self.change_samples is None:
                raise Exception("change_samples not set eiter provide as method argument or run est_pct_change")
            else:
                change_samples = self.change_samples
        if states is None:
            states = list(change_samples.index.levels[0])
        else:
            assert all([s in change_samples.index.levels[0] for s in states]),"No data for some states in provided list" 
            change_samples = change_samples.loc[states, : ].copy()
        if policies is None:
            policies = list(np.unique( [ x for y in change_samples.index.levels[1] for x in y.split("_AND_")] ) )
        ####### Helper functions
        def format_xlab(s, width = 30):
            s = "\nAND\n".join([textwrap.fill(x, width = width, initial_indent = '   ', subsequent_indent= '   '
                                              ) if i>0 else textwrap.fill(x, width = width
                                                                             ) for i,x in enumerate(s.split("_AND_")) ])
            return  s
        ####### PLOT
        nrows = len(policies)
        fig, axes = plt.subplots(figsize = (height*aspect, height*nrows), nrows = nrows , ncols = 1 )
        axes = np.ravel(axes)
        for p, ax in zip(policies, axes):
            mask = [ len(set(i[1].split("_AND_")).intersection([p])) > 0 for i in change_samples.index ]
            plot_data = change_samples.loc[mask, :].reset_index()
            plot_data["state/province_and_policy"] =  plot_data.apply(lambda x: x["state/province"] +" : " + x["policy"], axis = 1)
            plot_data = plot_data.drop(columns = ["state/province" , "policy" , "policy_date"])
            plot_data = plot_data.set_index("state/province_and_policy", verify_integrity=True)
            plot_data = plot_data.transpose().melt()
            ax = sns.boxplot(data = plot_data , x =  "state/province_and_policy" , y = "value", ax= ax, whis=[5, 95])
            ax.set_xticklabels( [format_xlab(x.get_text()) for x in  ax.get_xticklabels()], rotation = 90)
            ax.set_ylim( y_lim )
            ax.set_ylabel(y_label)
            ax.grid(True)
            ax.set_title(p)
        fig.subplots_adjust(hspace = hspace)
        return fig
    
    @staticmethod
    def _samples_to_cred_interval(samples, sample_axis = 1):
        """
        Construct a symmetric 95% credible interval (CI) (TODO: HDI  interval)
        Inputs
        ------
            samples — pd.DataFrame or pd.Series where one axis stores samples that are to be aggregated to CI
            samples_axis- int 
        """
        if isinstance(samples, pd.Series):
            out = pd.Series( [ samples.mean(), samples.quantile(0.025), samples.quantile(0.975)]  , index = ["mean" , "lower" , "upper"] )
        elif isinstance(samples, pd.DataFrame ):
            out = pd.concat([ samples.mean(axis = sample_axis),
                            samples.quantile(0.025, axis = sample_axis), 
                            samples.quantile(0.975, axis = sample_axis)
                            ],
                            axis = sample_axis, keys = ["mean" , "lower", "upper"] )
        else:
            raise ValueError()
        return out