import numpy as np 
import pandas as pd 

import pymc3 as pm
from scipy.linalg import toeplitz
import theano.tensor as tt


class InfectionSeriesEstimator(object):
    """
    Properties:
            P_e_given_i - (2d array) P_e_given_i[e,i] gives probability of event on day e given infection on day i.
            T_period - (2 tuple) — (min_days_elapsed_from_infect_to_transmission, max_days_elapsed_from_infect_to_transmission)
                                       (lower end inclusive, upper end exclusive)
            mu — exponential prefactor in prior
            inputs — dictionary of inputs
            D - Matrix for calculating 1d discrete difference in log(tranmission_rataes)  
    Basic Usage:
        ise = Infection_Series_Estimator(event_series =deaths_by_day[state].copy().astype(int), 
                                            P_e_given_i= make_infect_to_death_pmf(), 
                                            T_period = T_period,
                                            policy_dates= policy_dates )
        p_i, Re, p_i_samples, Re_samples= ise.fit(...)
        p_e, p_e_samples, n_e, n_e_samples = ise.predict_p_e()
    """
    def __init__(self, event_series, P_e_given_i, T_period, policy_dates):
        """ 
        Inputs:
            event_series — pd.Series - index is dates values are counts
            P_e_given_i - 1d array repreenting pmf for event occuring k = (e-i) days after infection
                        
            T_period - (2 tuple) — (min_days_elapsed_from_infect_to_transmission, max_days_elapsed_from_infect_to_transmission)
                                       (lower end inclusive, upper end exclusive)
            rms_foldchange — root mean square log2 ratio of tranmission rates on sequential dates. Determines strength of pior
        """
        self.init_kwargs = { "event_series" :event_series,
                        "P_e_given_i": P_e_given_i,
                       "T_period" : T_period,
                       "policy_dates" : policy_dates}
        ## counts and date info
        self.event_series = self._preprocess_event_series(event_series, P_e_given_i)
        date_max = self.event_series.index[-1]
        date_min = self.event_series.index[0]
        self.days_total = (date_max - date_min).days + 1
        
        if T_period[0] != 1:
            raise ValueError("Code only supports models with transmission period beginning 1 d after infection")
        self.T_period = T_period
        
        policy_dates_processed = self._preprocess_policy_dates(policy_dates)
        self.policy_dates = [ ( (x[0]-date_min).days, (x[1]-date_min).days) for x in  policy_dates_processed ]
        
    @staticmethod
    def _preprocess_event_series(event_series, P_e_given_i_1d):
        """
        Insures event series is:
            - sorted by increasing date
            - has timestamp index
            - starts at at least len(P_e_given_i_1d) -1 before first day with event_series >0
        
        Returns
             event_series
        """
        ## Sort and check index type
        event_series = event_series.sort_index().copy()
        if not isinstance(event_series.index, pd.core.indexes.datetimes.DatetimeIndex):
            event_series.index = pd.to_datetime(event_series.index)
        ## Truncate or extend index
        earliest_nonzero = (event_series > 0).idxmax() 
        date_min = earliest_nonzero - pd.Timedelta(len(P_e_given_i_1d)-1, unit = "days")                  
        if date_min < event_series.index[0]:
            prepend_series =  pd.Series(index = pd.date_range(start = date_min,
                                                              end =event_series.index[0],
                                                               freq = 'D', closed = 'left' ),
                                        data = 0)     
            event_series = pd.concat( [prepend_series, event_series ] )    
        else:
            event_series = event_series.loc[date_min:].copy()
        return event_series
    
    @staticmethod
    def _preprocess_policy_dates(policy_dates):
        out = []
        for elem in policy_dates:
            if isinstance(elem, str):
                out.append( ( pd.Timestamp(elem), pd.Timestamp(elem) + pd.Timedelta(1, unit = "D") ) )
            elif isinstance(elem, tuple) or isinstance(elem, list):
                if len(elem) == 1:
                    out.append( ( pd.Timestamp(elem[0]), pd.Timestamp(elem[0]) + pd.Timedelta(1, unit = "D") ) )
                elif len(elem) == 2:
                    out.append( tuple( pd.Timestamp(x) for x in elem ) )
                else:
                    raise ValueError("At least one of the entries of policy_dates has length > 2!")
            else:
                raise ValueError("Could not parse entry {} of policy_dates".format(elem ))
        return out
    
    @property
    def P_e_given_i(self):
        P_e_given_i = self._make_P_e_given_i(self.init_kwargs["P_e_given_i"], self.days_total, observed_only = True)
        return  P_e_given_i
    @property
    def M1(self):
        return None
    @property
    def M2(self):
        return None
    @property
    def D(self):
        return None
    
    @staticmethod
    def _make_P_e_given_i(P_e_given_i, days_total, observed_only = True):
        """
        P_e_given_i - 1d numpy array
        """
        if observed_only:
            P_e_given_i = toeplitz(c = np.concatenate([ P_e_given_i, np.zeros(days_total-len(P_e_given_i)) ] ),
                                   r =  np.concatenate([ P_e_given_i[0:1], np.zeros(days_total- 1) ] ), 
                                  )
        else:
            P_e_given_i = toeplitz(c = np.concatenate([ P_e_given_i, np.zeros(days_total-1) ] ),
                                   r =  np.concatenate([ P_e_given_i[0:1], np.zeros(days_total- 1) ] ), 
                                  )
        return P_e_given_i 
  
    @staticmethod
    def _make_pior_mats(T_period, days_total):
        """
        
        """
        M1 =  toeplitz( c = np.zeros( days_total-(T_period[1]-T_period[0]) ) ,
                        r = np.concatenate([ np.zeros(T_period[1]-T_period[0]),
                                             np.array([1.]) , 
                                             np.zeros( days_total - 1 - (T_period[1]-T_period[0]) ), 
                                           ])
                      )
        M2 =  toeplitz( c = np.concatenate([ np.array([1.]), np.zeros( days_total-(T_period[1]-T_period[0]) -1 ) ] ) ,
                        r = np.concatenate([ np.ones(T_period[1]-T_period[0]),
                                            np.zeros(days_total - (T_period[1]-T_period[0])) ] )
                      )
        assert M1.shape == M2.shape
        D =  toeplitz( c = np.concatenate([ np.array([-1.]), np.zeros(M1.shape[0]-2)  ]),
                       r = np.concatenate([ np.array([-1., 1]), np.zeros(M1.shape[0]-2) ] )
                     ) 
        return M1, M2, D
    
    def fit(self, mu, policy_factor, test_val = None,  **kwargs):
        
        P_e_given_i = self._make_P_e_given_i(self.init_kwargs["P_e_given_i"], self.days_total)
        M1, M2, D = self._make_pior_mats(self.T_period, self.days_total)
        if test_val is None:
            test_val = np.ones(len(self.event_series))/len(self.event_series)
        
        p_i_samples, self.posterior = self._sample_posterior( counts_arr = self.event_series.values.copy() , 
                                                              P_e_given_i = P_e_given_i, 
                                                              M1 = M1,
                                                              M2 = M2, 
                                                              D = D, 
                                                              mu = mu, 
                                                              days_total = self.days_total, 
                                                              test_val = test_val,
                                                              policy_dates = self.policy_dates, 
                                                              policy_factor = policy_factor, 
                                                             **kwargs)
        self.p_i_samples = pd.DataFrame( index = self.event_series.index ,
                                         data =  p_i_samples.transpose() )
        self.p_i = self._samples_to_cred_interval( self.p_i_samples,  sample_axis =1  )
        
        ## caculate R_e
        self.Re_samples = self._est_Re(self.p_i_samples, self.T_period )
        self.Re = self._samples_to_cred_interval(self.Re_samples ,  sample_axis =1  )
    
        return  self.p_i.copy(), self.Re.copy(), self.p_i_samples.copy(), self.Re_samples.copy() 
                                 
    @staticmethod
    def _sample_posterior(counts_arr, P_e_given_i, M1, M2, D, mu, days_total, test_val,
                          policy_dates, policy_factor, **kwargs):
        
        n = int(counts_arr.sum())
        ## multiply mu by policy_factor to allow more change on policy_dates
        mu_hard = mu*np.ones(D.shape[0])
        mu_soft = mu_hard.copy()
        policy_day_shift =  np.argmax( M1[0,:]) + 1
        for start, stop in policy_dates:
            mu_soft[ start-policy_day_shift :stop-policy_day_shift] *= policy_factor
        
        with pm.Model() as model:
            p_i = InfectSeriesPrior("p_i",
                                       mu_soft=mu_soft ,
                                       mu_hard = mu_hard, 
                                       M1=M1, 
                                       M2 =M2, 
                                       D= D, 
                                       G = P_e_given_i,
                                       N =n,
                                       testval = test_val, 
                                       shape= days_total,
                                       transform=pm.distributions.transforms.stick_breaking)
            p_e = tt.dot(P_e_given_i, p_i ) 
            p_e_given_observed =  p_e/ tt.sum(p_e)
            N = pm.Multinomial("N", n=n, p=p_e_given_observed, observed= counts_arr)
               
            posterior = pm.sample(cores= 1, **kwargs) 
            
        return posterior["p_i"].copy(), posterior
    
    @staticmethod
    def _est_Re(p_i_samples, T_period,  min_infected_frac= 0.0001 ):
        """
        p_i_samples  - DataFrame with 
                                values — proportional to number of infected people 
                                index — dates, 
                                columns — posterior samples
        T_period — 2-tuple
        min_infected_frac — float                 
        trans_period  - 2-tuple representing left end open right end closed interval
        """
        Re_idx_min =  max(T_period[1]-1, np.argmax(p_i_samples.mean(axis = 1).values > min_infected_frac) )
        Re_samples = pd.DataFrame(index = p_i_samples.index[Re_idx_min:].copy(), 
                                 columns = p_i_samples.columns.copy(),
                                 data = 0. )
        for idx in range(Re_idx_min, p_i_samples.shape[0]):
            Re_samples.iloc[idx - Re_idx_min, :] = (p_i_samples.iloc[idx, :]
                                                    )/( (p_i_samples.iloc[idx - T_period[1]+1 : idx - T_period[0] + 1, :]
                                                                ).sum(axis = 0) 
                                                       )
        Re_samples= Re_samples*(T_period[1] - T_period[0])
        return Re_samples
    
    
    def predict_p_e(self,):
        
        P_e_given_i = InfectionSeriesEstimator._make_P_e_given_i(self.init_kwargs["P_e_given_i"], self.days_total, observed_only = False )
        p_e_samples = P_e_given_i@(self.p_i_samples.values)
        
        p_e_samples = pd.DataFrame(data = p_e_samples, columns = self.p_i_samples.columns)
        p_e_index_observed = self.p_i_samples.index.copy()
        p_e_index_unobserved = p_e_index_observed[-1] + pd.timedelta_range( start = pd.Timedelta(days = 1), 
                                                                             end =   pd.Timedelta(days = len(p_e_samples) - len(p_e_index_observed) ),
                                                                              freq = 'd' )
        p_e_index = p_e_index_observed.append(p_e_index_unobserved)          
        p_e_samples.index= p_e_index 
        p_e = self._samples_to_cred_interval(p_e_samples ,  sample_axis =1  )
        
        ## Expected number of events per day
        n_observed = self.event_series.sum()
        n_e_samples = p_e_samples.apply(
                            lambda x: x*n_observed*np.concatenate([ np.ones(len(p_e_index_observed))/x.loc[p_e_index_observed].sum(),
                                                                    (1./x.loc[p_e_index_observed].sum()
                                                                        )*np.ones(len(p_e_index_unobserved))
                                                              ] ),
                            axis = 0)
        n_e = self._samples_to_cred_interval(n_e_samples, sample_axis =1  )
                 
        return p_e,  p_e_samples, n_e, n_e_samples
    
    @staticmethod
    def _samples_to_cred_interval(samples , sample_axis =1 ):

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
    
class InfectSeriesPrior( pm.Continuous):
    def __init__(self, mu_soft, mu_hard, M1, M2, D, G, N, *args, **kwargs):
        super(InfectSeriesPrior, self).__init__(*args, **kwargs)
        
        self.mu_soft = mu_soft
        self.mu_hard = mu_hard
        self.M1 = M1
        self.M2 = M2
        self.D = D
        self.G = G
        self.N = N
        
    def logp(self, value):
        mu_soft = self.mu_soft
        mu_hard = self.mu_hard
        M1 = self.M1
        M2 = self.M2
        D = self.D
        G = self.G
        N = self.N
        
        diff = tt.dot(D, tt.log(tt.dot(M1,value)) - tt.log(tt.dot(M2,value)))
        t_out = -1.*tt.sum( tt.switch(tt.lt(diff, 0.), mu_soft, mu_hard )*diff**2  )
                          
        return t_out
    
