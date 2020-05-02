import numpy as np 
import pandas as pd 
import scipy as sp
import pymc3 as pm
import theano.tensor as tt

from datetime import timedelta
# from sklearn.mixture import GaussianMixture
# import progressbar

import matplotlib.colors as colors
import matplotlib.pyplot as plt  

class LapDerivative(pm.Continuous):
    def __init__(self, pen, M1, M2, D, *args, **kwargs):
        super(LapDerivative, self).__init__(*args, **kwargs)
        
        self.pen = pen
        self.M1 = M1
        self.M2 = M2
        self.D = D
        
    def logp(self, value):
        D = self.D
        M1 = self.M1
        M2 = self.M2
        pen = self.pen
        t_out = -pen*tt.sum( 
                        tt.dot(D ,
                            tt.log(tt.dot(M1,value)) -  tt.log(tt.dot(M2,value))   
                                  )**2
                        )
        return t_out

class Infect_Date_Estimator(object):
    def __init__(self, event_series, like_func, policy_dates, trans_period = (1,8)  ):
        """
        event_series -  index is of type datetime64[ns]
        like_func - pmf for likelihood of event (death/hospitalization) as function of days after infection
        """

        self.event_series = self._preprocess_event_series(event_series, like_func)
        self.date_earliest = self.event_series.index[0]
        self.date_latest = self.event_series.index[-1]
        self.n_days = (self.date_latest - self.date_earliest).days + 1
        policy_dates_processed = self._preprocess_policy_dates(policy_dates)
        self.policy_dates = [ ( (x[0]-self.date_earliest).days, (x[1]-self.date_earliest).days) for x in  policy_dates_processed  ]
        
        self.like_func = like_func
        self._like_func_Nzero = np.argmax(like_func > 0.)
        self.N_obsCas = self.event_series.sum()
        if trans_period[0] != 1:
            raise ValueError("Code only supports models with transmission period beginning 1 d after infection")
        self.trans_period = trans_period
        
        self.infect_count_cas = None
        self.infect_count_all = None
        self.p_i_from_boot = None

    @property
    def conditional_prob_mat(self,):
        cond_probs = self._make_cond_prob_mat(self.like_func, self.n_days, given_obsCas=True)
        cond_probs =  pd.DataFrame( data= cond_probs, 
                                    index = self.event_series.index ,
                                    columns = self.event_series.index.shift(-1*self._like_func_Nzero, freq='D') ) 
        return cond_probs
    @property
    def log_penalty_mats(self):
        M1 , M2= Infect_Date_Estimator._make_log_penalty_mats(self.n_days, window_size=self.trans_period[1]-1, like_func = self.like_func)
        return M1, M2
    @property 
    def D1(self):
        D1 = Infect_Date_Estimator._make_d1_mat(n_days = self.n_days,
                                                like_func=self.like_func,
                                                trans_period = self.trans_period,
                                                 policy_dates= self.policy_dates )
        return D1
    @property
    def N_i_obsCas_expected(self):
        return  self.p_i_given_obsCas*self.N_obsCas
    @property
    def Ro(self):
        Ro ,_ = self._est_Ro( infected_counts = self.samples_p_i.copy(), trans_period = self.trans_period,  min_infected_frac =0.0001)
        return  Ro
    @property
    def samples_Ro(self):
        _ , samples_Ro = self._est_Ro( infected_counts = self.samples_p_i.copy(), trans_period = self.trans_period, min_infected_frac =0.0001)
        return samples_Ro
    
    @staticmethod
    def _preprocess_policy_dates(policy_dates):
        out = []
        for elem in policy_dates:
            if isinstance(elem, str):
                out.append( ( pd.Timestamp(elem), pd.Timestamp(elem) + timedelta(days=1) ) )
            elif isintance(elem, tuple) or isintance(elem, list):
                if len(elem) == 1:
                    out.append( ( pd.Timestamp(elem[0]), pd.Timestamp(elem[0]) + timedelta(days=1) ) )
                elif len(elem) == 2:
                    out.append( tuple( pd.Timestamp(x) for x in elem ) )
                else:
                    raise ValueError("At least one of the entries of policy_dates has length > 2!")
            else:
                raise ValueError("Could not parse entry {} of policy_dates".format(elem ))
        return out

    @staticmethod
    def _preprocess_event_series(event_series, like_func):
        event_series = event_series.sort_index().copy()
        date_earliest = event_series.index[0]
        d_preceeding  = len(like_func[np.argmax( like_func > 0.): ]) - 1
        prepend_series = pd.Series(index =  date_earliest + \
                                               pd.timedelta_range(start = timedelta(days = -1*d_preceeding), 
                                                                    end = timedelta(days = -1), 
                                                                    freq = 'd'
                                                                ),
                                        data = 0.
                                      )
        event_series = pd.concat( [prepend_series, event_series ] ) ## prepend dates
        return event_series

    @staticmethod
    def _samples_to_cred_interval( samples , sample_axis =1 ):

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

    def est_infect_dates(self, smooth_penalty, flex_factor=0.01, **kwargs_pm_sample ): 

        p_i_given_obsCas, self.posterior_samples = self._sample_posterior( event_counts = self.event_series.values.copy(), 
                                                                            like_func = self.like_func,
                                                                            trans_period= self.trans_period,
                                                                            policy_dates = self.policy_dates,
                                                                            penalty = smooth_penalty,
                                                                            flex_factor = flex_factor ,
                                                                            **kwargs_pm_sample )
        ## Organize results
        p_i_index = self.event_series.index.shift(-1*self._like_func_Nzero, freq='D')  ### shift index for infecation dates
        self.samples_p_i_given_obsCas = pd.DataFrame(index= p_i_index,  data = p_i_given_obsCas.transpose())
        self.p_i_given_obsCas = self._samples_to_cred_interval(self.samples_p_i_given_obsCas, sample_axis =1 )

        self.samples_p_i, self._samples_p_i_full, self._fracs_expected = self._rescale_p_i( self.samples_p_i_given_obsCas,
                                                                                             self.like_func, 
                                                                                             self.n_days)    
        self.p_i =  self._samples_to_cred_interval(self.samples_p_i,)                                                                      

        self._N_cas_full = self._samples_to_cred_interval( self.N_obsCas/(1. - self._fracs_expected) )
        self._samples_N_cas = ( self.N_obsCas/(1.-self._fracs_expected))*self._samples_p_i_full.loc[self.p_i.index, :].sum(axis = 0)
        self.N_cas = self._samples_to_cred_interval( self._samples_N_cas )

        self.samples_p_e = self._calc_p_e(self.samples_p_i.copy(), self.like_func.copy() )
        self.p_e = self._samples_to_cred_interval( self.samples_p_e  )
        self._samples_N_cas_by_day = self.samples_p_e.multiply( self._samples_N_cas, axis = 1 )
        self.N_cas_by_day = self._samples_to_cred_interval( self._samples_N_cas_by_day  )
        
        return self.p_i_given_obsCas , self.p_i, self.N_cas_by_day

    @staticmethod
    def _rescale_p_i(samples_p_i_given_obsCas, like_func, n_days):
        """
        Calculates probabilites of infection on certain day from probabilies of infection on that day condition on observed death.
        Also Calculates the fraction of infections that have occurred so far for which casualties are expected.
        """
        like_func = like_func[np.argmax(like_func>0.) :].copy()
        scale_factor = np.array( [ np.sum(like_func[:n_days-d]) for d in range(n_days) ] )

        p_i_full = samples_p_i_given_obsCas.divide( scale_factor, axis = 0 )
        frac_expected = 1. - 1./p_i_full.sum(axis = 0)
        p_i_full = p_i_full.divide( p_i_full.sum(axis = 0), axis = 1 )

        d_truncate = np.argmax(np.cumsum(like_func) > 0.05)
        p_i =  (p_i_full.iloc[:-1*d_truncate,:]).divide(  p_i_full.iloc[:-1*d_truncate,:].sum(axis = 0), axis =1  )

        return p_i, p_i_full, frac_expected
    
    @staticmethod
    def _calc_p_e(p_i, like_func,):
        cond_probs = Infect_Date_Estimator._make_cond_prob_mat(like_func, p_i.shape[0], given_obsCas = False)
        p_e = cond_probs@(p_i.values)
        
        p_e = pd.DataFrame(data = p_e, columns = p_i.columns)
        p_e_index = p_i.index.copy()
        p_e_index = p_e_index.append( p_e_index[-1] + pd.timedelta_range( start = timedelta(days = 1), 
                                                                        end = timedelta(days = len(like_func[np.argmax(like_func>0.) :]) -1), 
                                                                        freq = 'd'  ) 
                                     )
        p_e_index = p_e_index.shift(np.argmax(like_func>0),  freq='D')  ### shift index to account leading 0 of conditional prob p(death = i | infect = j)
        p_e.index= p_e_index 
        return p_e

    @staticmethod
    def _make_log_penalty_mats(size, window_size, like_func):
        like_func = like_func[np.argmax(like_func>0.) :].copy()
        
        m1 = np.zeros( (size - window_size, size), dtype = float )
        for j in range(window_size, size):
            if size - j >= len(like_func):
                m1[j - window_size, j ] = 1.
            else:
                 m1[j - window_size, j ] =1./np.sum(like_func[: size -j])
        m2 = np.zeros( (size - window_size, size), dtype = float )
        for i in range(m2.shape[0]):
            for j in range(m2.shape[1]):
                if j >=i:
                    if j < i+window_size:
                        m2[i,j] = 1./np.sum(like_func[:size-j])
                    else:
                        m2[i,j] = 0.
                else:
                    m2[i,j] = 0.
        return m1, m2
                
    @staticmethod
    def _make_cond_prob_mat(like_func, n_days, given_obsCas= True ):
        """
        like_func - 1d array
        n_days - int (the number of days with event data)
        """
        if  given_obsCas:
            like_func_nonzero = like_func[np.argmax(like_func>0.) :].copy()
            cond_probs = np.zeros( shape =(n_days, n_days), dtype = float )
            for j in range(n_days):
                max_days_rel = min( len(like_func_nonzero), n_days-j )
                cond_probs[j:j+max_days_rel,j] =  like_func_nonzero[:max_days_rel] / np.sum(like_func_nonzero[:max_days_rel])
        else:
            like_func_nonzero = like_func[np.argmax(like_func>0.) :].copy()
            cond_probs = np.zeros( shape =(n_days+len(like_func_nonzero) -1, n_days ), dtype = float)
            for j in range(n_days):
                cond_probs[j:j+len(like_func_nonzero),j] = like_func_nonzero
        return cond_probs

    @staticmethod
    def _sample_posterior(event_counts, like_func, trans_period, policy_dates, penalty, flex_factor, **kwargs ):
        n_days = len(event_counts)
        cond_probs = Infect_Date_Estimator._make_cond_prob_mat(like_func, n_days)
        
        A = cond_probs.copy()
        b=  event_counts.copy()
        M1 ,M2 = Infect_Date_Estimator._make_log_penalty_mats(n_days, window_size=trans_period[1]-1, like_func = like_func)
        D = Infect_Date_Estimator._make_d1_mat( n_days = n_days, 
                                                like_func= like_func,
                                                policy_dates = policy_dates,
                                                trans_period= trans_period, 
                                                flex_factor= flex_factor )

        with pm.Model() as model:
            theta = LapDerivative("theta",
                                pen = penalty, 
                                M1 =M1,
                                M2 = M2,
                                D= D,
                                testval = np.ones(n_days)/n_days, shape= n_days,
                                transform=pm.distributions.transforms.stick_breaking )
            p = tt.dot(A, theta )
            N = pm.Multinomial("N", n=event_counts.sum(), p=p, observed= b)
            
            posterior = pm.sample(cores= 1, **kwargs)
        
        return posterior["theta"].copy(), posterior

    @staticmethod
    def _fit_MAP(event_counts, like_func, trans_period , penalty = None ):
        """
        
        """
        n_days = len(event_counts)
        cond_probs = Infect_Date_Estimator._make_cond_prob_mat(like_func, n_days)

        A = cond_probs.copy()
        b=  event_counts
        M1 ,M2 = Infect_Date_Estimator._make_log_penalty_mats(n_days, window_size=trans_period[1]-1, like_func = like_func)
        D = Infect_Date_Estimator._make_d1_mat( n_days, like_func = like_func, trans_period = trans_period )

        p_count = 0.00005
        def obj_func(x, A, b, M1, M2, D, penalty):
            obj = -b@np.log( np.abs( A@x + p_count) ) + \
                    penalty*np.sum(( D@( np.log(np.abs(M1@x + p_count )) - np.log(np.abs(M2@x + p_count)) )
                                    )**2 )
            return obj

        def jac_func(x,A, b, M1, M2, D, penalty):
            A_x = A@x
            M1_x = M1@x
            M2_x = M2@x
            jac_ls = -b@np.diag( np.sign(A_x+p_count)/np.abs(A_x+p_count))@A   
            jac_pen = ( np.log(np.abs(M1_x+p_count)) - np.log(np.abs(M2_x+p_count))
                                    )@(np.transpose(D)@D
                                        )@( np.diag(np.sign(M1_x+p_count)/np.abs(M1_x+p_count))@M1 - \
                                            np.diag(np.sign(M2_x+p_count)/np.abs(M2_x+p_count))@M2 )

            return jac_ls + 2.*penalty*jac_pen

        opt_res =  sp.optimize.minimize( fun = obj_func, 
                                        x0 = np.ones(A.shape[1])/A.shape[1] , 
                                        jac = jac_func , 
                                        args = (A, b, M1, M2, D, penalty),
                                        method = 'trust-constr',
                                        bounds = [(0.,1.)]* A.shape[1] ,
                                        constraints = sp.optimize.LinearConstraint( np.ones( (1,A.shape[1]), dtype = float) ,
                                                                                    lb = 1., ub = 1. )
                                        )
        if not opt_res.success:
                raise ValueError("sp.optimize.minimize failed to converge with message:\n{}".format(opt_res.message) )
        else:
            print(opt_res.message)

        return opt_res.x, opt_res.fun, None

    @staticmethod
    def _est_Ro(infected_counts, trans_period, min_infected_frac= 0.0001 ):
        """
        infected_counts  - DataFrame with 
                                values — proportional to number of infected people 
                                index — dates, 
                                columns — posterior samples
        trans_period  — 2-tuple
        min_infected_frac — float                 
        trans_period  - 2-tuple representing left end open right end closed interval
        """
        est_start_idx = max( trans_period[1] - 1, np.argmax( infected_counts.values >=min_infected_frac) )
        samples_Ro = pd.DataFrame(index = infected_counts.index[est_start_idx:].copy(), 
                                       columns = infected_counts.columns.copy(),
                                       data = 0. )
        for idx in range(est_start_idx, len(infected_counts) ):
            samples_Ro.iloc[idx - est_start_idx, :] = infected_counts.iloc[idx, :] / ( 
                                                            (infected_counts.iloc[idx - trans_period[1]+1 : idx - trans_period[0] + 1, :]
                                                                ).sum(axis = 0)  )
        samples_Ro= samples_Ro*(trans_period[1] - trans_period[0])
        Ro = Infect_Date_Estimator._samples_to_cred_interval(samples_Ro)
        return Ro, samples_Ro
 
    @staticmethod
    def _make_d2_mat(size):
        """
        size - int — the number of rows and columns in matrix
        """
        d2 = np.zeros((size,size), dtype = float)
        for i in range(size):
            if i+2 < size:
                 d2[i,i:i+3] = [1.,-2.,1.]
        return d2

    @staticmethod
    def _make_d1_mat(n_days, like_func, trans_period, policy_dates = [], flex_factor = 0.01):
        size = n_days-(trans_period[1] -1)
        shift = np.argmax(like_func>0.) - (trans_period[1]-1) -1  ## shift accounts for: date shift due to 0 probs at 
                                                            ## start of liklihood function, full transmission period must elapse before 
                                                        ## calculating transmission rates, and -1 to select the row comparing the preceeding
                                                        ## date with date of interest
        p_dates = [ (x[0] + shift , x[1] + shift) for x in policy_dates ]
        d1 = np.zeros((size-1,size), dtype = float)
        for i in range(0,size-1):
            d1[i,i:i+2] = [-1., 1.]
        for start, stop in  p_dates:
            d1[ start: stop, :  ] *= flex_factor
        return d1

    def est_total_infect_by_day(self, age_group_data):
        """
        """
        self.infect_count_all =  self.infect_count_cas / (age_group_data["mortality_rate"]*age_group_data["pop_frac"]).sum()
        return self.infect_count_all.copy() 



# class Infect_Date_Estimator_OLD(object):
#     def __init__(self, event_series, like_func, trans_period = (1,8)  ):
#         """
#         event_series -  index is of type datetime64[ns]
#         like_func - pmf for likelihood of event (death/hospitalization) as function of days after infection
#         """

#         self.event_series = self._preprocess_event_series(event_series, like_func)
#         self.date_earliest = self.event_series.index[0]
#         self.date_latest = self.event_series.index[-1]
#         self.n_days = (self.date_latest - self.date_earliest).days + 1
        
#         self.like_func = like_func
#         self._like_func_Nzero = np.argmax(like_func > 0.)
#         self.N_obsCas = self.event_series.sum()
#         if trans_period[0] != 1:
#             raise ValueError("Code only supports models with transmission period beginning 1 d after infection")
#         self.trans_period = trans_period
        
#         self.infect_count_cas = None
#         self.infect_count_all = None
#         self.p_i_from_boot = None

#     @property
#     def conditional_prob_mat(self,):
#         cond_probs = self._make_cond_prob_mat(self.like_func, self.n_days, given_obsCas=True)
#         cond_probs =  pd.DataFrame( data= cond_probs, 
#                                     index = self.event_series.index ,
#                                     columns = self.event_series.index.shift(-1*self._like_func_Nzero, freq='D') ) 
#         return cond_probs

#     @property
#     def log_penalty_mats(self):
#         M1 , M2= Infect_Date_Estimator._make_log_penalty_mats(self.n_days, window_size=self.trans_period[1]-1, like_func = self.like_func)
#         return M1, M2
#     @property 
#     def D1(self):
#         D1 = Infect_Date_Estimator._make_d1_mat( self.n_days-(self.trans_period[1] -1) )
#         return D1

#     @staticmethod
#     def _preprocess_event_series(event_series, like_func):
#         event_series = event_series.sort_index().copy()
#         date_earliest = event_series.index[0]
#         d_preceeding  = len(like_func[np.argmax( like_func > 0.): ]) - 1
#         prepend_series = pd.Series(index =  date_earliest + \
#                                                pd.timedelta_range(start = timedelta(days = -1*d_preceeding), 
#                                                                     end = timedelta(days = -1), 
#                                                                     freq = 'd'
#                                                                 ),
#                                         data = 0.
#                                       )
#         event_series = pd.concat( [prepend_series, event_series ] ) ## prepend dates
#         return event_series

#     def est_casualty_infect_dates(self, smooth_penalty =None  ): 

#         p_i_given_obsCas, loss , _ = self._fit( event_counts = self.event_series.values.copy(), 
#                                                         like_func = self.like_func,
#                                                         penalty = smooth_penalty,
#                                                         trans_period= self.trans_period)
                    
#         ## Organize results
#         p_i_index = self.event_series.index.shift(-1*self._like_func_Nzero, freq='D')  ### shift index for infecation dates
#         self.p_i_given_obsCas = pd.Series( index= p_i_index,  data = p_i_given_obsCas)
#         self.N_i_obsCas_expected = self.p_i_given_obsCas*self.N_obsCas
        
#         self.p_i, self._p_i_full, self._frac_expected = self._rescale_p_i(self.p_i_given_obsCas, self.like_func, self.n_days)
#         self._N_cas_full = self.N_obsCas/(1. - self._frac_expected) 
#         self.N_cas = self._N_cas_full*(self._p_i_full.loc[self.p_i.index].sum())
        
#         self.p_e = self._calc_p_e(self.p_i.copy(), self.like_func.copy() )
#         self.N_i_cas_expected = self.p_e * self.N_cas
#         return self.p_i_given_obsCas, self.p_i, self.N_i_cas_expected 

#     @staticmethod
#     def _make_cond_prob_mat(like_func, n_days, given_obsCas= True ):
#         """
#         like_func - 1d array
#         n_days - int (the number of days with event data)
#         """
#         if  given_obsCas:
#             like_func_nonzero = like_func[np.argmax(like_func>0.) :].copy()
#             cond_probs = np.zeros( shape =(n_days, n_days), dtype = float )
#             for j in range(n_days):
#                 max_days_rel = min( len(like_func_nonzero), n_days-j )
#                 cond_probs[j:j+max_days_rel,j] =  like_func_nonzero[:max_days_rel] / np.sum(like_func_nonzero[:max_days_rel])
#         else:
#             like_func_nonzero = like_func[np.argmax(like_func>0.) :].copy()
#             cond_probs = np.zeros( shape =(n_days+len(like_func_nonzero) -1, n_days ), dtype = float)
#             for j in range(n_days):
#                 cond_probs[j:j+len(like_func_nonzero),j] = like_func_nonzero
#         return cond_probs

#     @staticmethod
#     def _calc_p_e(p_i, like_func,):
#         cond_probs = Infect_Date_Estimator._make_cond_prob_mat(like_func, len(p_i), given_obsCas = False)
#         p_e = cond_probs@(p_i.values)
        
#         p_e = pd.Series(data = p_e)
#         p_e_index = p_i.index.copy()
#         p_e_index = p_e_index.append( p_e_index[-1] + pd.timedelta_range( start = timedelta(days = 1), 
#                                                                         end = timedelta(days = len(like_func[np.argmax(like_func>0.) :]) -1), 
#                                                                         freq = 'd'  ) 
#                                      )
#         p_e_index = p_e_index.shift(np.argmax(like_func>0),  freq='D')  ### shift index to account leading 0 of conditional prob p(death = i | infect = j)
#         p_e.index= p_e_index 
#         return p_e

#     @staticmethod
#     def _rescale_p_i(p_i_given_obsCas, like_func, n_days, ):
#         """
#         Calculates probabilites of infection on certain day from probabilies of infection on that day condition on observed death.
#         Also Calculates the fraction of infections that have occurred so far for which casualties are expected.
#         """
#         like_func = like_func[np.argmax(like_func>0.) :].copy()
#         p_i_full = pd.Series(index = p_i_given_obsCas.index, 
#                         data = [ x/np.sum(like_func[:n_days-d]) for d,x in enumerate(p_i_given_obsCas.values) ])
#         frac_expected = 1. - 1./p_i_full.sum()
#         p_i_full = p_i_full/p_i_full.sum()
        
#         d_truncate = np.argmax(np.cumsum(like_func) > 0.05)
#         p_i =  p_i_full.iloc[:-1*d_truncate]/ p_i_full.iloc[:-1*d_truncate].sum()
#         return p_i, p_i_full, frac_expected
    
#     @staticmethod
#     def _make_log_penalty_mats(size, window_size, like_func):
#         like_func = like_func[np.argmax(like_func>0.) :].copy()
        
#         m1 = np.zeros( (size - window_size, size), dtype = float )
#         for j in range(window_size, size):
#             if size - j >= len(like_func):
#                 m1[j - window_size, j ] = 1.
#             else:
#                  m1[j - window_size, j ] =1./np.sum(like_func[: size -j])
#         m2 = np.zeros( (size - window_size, size), dtype = float )
#         for i in range(m2.shape[0]):
#             for j in range(m2.shape[1]):
#                 if j >=i:
#                     if j < i+window_size:
#                         m2[i,j] = 1./np.sum(like_func[:size-j])
#                     else:
#                         m2[i,j] = 0.
#                 else:
#                     m2[i,j] = 0.
#         return m1, m2
                
#     @staticmethod
#     def _fit(event_counts, like_func, trans_period , penalty = None ):
#         """
        
#         """
#         n_days = len(event_counts)
#         cond_probs = Infect_Date_Estimator._make_cond_prob_mat(like_func, n_days)

#         A = cond_probs.copy()
#         b=  event_counts
#         M1 ,M2 = Infect_Date_Estimator._make_log_penalty_mats(n_days, window_size=trans_period[1]-1, like_func = like_func)
#         D = Infect_Date_Estimator._make_d1_mat( n_days-(trans_period[1] -1) )

#         p_count = 0.00005
#         def obj_func(x, A, b, M1, M2, D, penalty):
#             obj = -b@np.log( np.abs( A@x + p_count) ) + \
#                     penalty*np.sum(( D@( np.log(np.abs(M1@x + p_count )) - np.log(np.abs(M2@x + p_count)) )
#                                     )**2 )
#             return obj

#         def jac_func(x,A, b, M1, M2, D, penalty):
#             A_x = A@x
#             M1_x = M1@x
#             M2_x = M2@x
#             jac_ls = -b@np.diag( np.sign(A_x+p_count)/np.abs(A_x+p_count))@A   
#             jac_pen = ( np.log(np.abs(M1_x+p_count)) - np.log(np.abs(M2_x+p_count))
#                                     )@(np.transpose(D)@D
#                                         )@( np.diag(np.sign(M1_x+p_count)/np.abs(M1_x+p_count))@M1 - \
#                                             np.diag(np.sign(M2_x+p_count)/np.abs(M2_x+p_count))@M2 )

#             return jac_ls + 2.*penalty*jac_pen

#         opt_res =  sp.optimize.minimize( fun = obj_func, 
#                                         x0 = np.ones(A.shape[1])/A.shape[1] , 
#                                         jac = jac_func , 
#                                         args = (A, b, M1, M2, D, penalty),
#                                         method = 'trust-constr',
#                                         bounds = [(0.,1.)]* A.shape[1] ,
#                                         constraints = sp.optimize.LinearConstraint( np.ones( (1,A.shape[1]), dtype = float) ,
#                                                                                     lb = 1., ub = 1. )
#                                         )
#         if not opt_res.success:
#                 raise ValueError("sp.optimize.minimize failed to converge with message:\n{}".format(opt_res.message) )
#         else:
#             print(opt_res.message)

#         return opt_res.x, opt_res.fun, None
    
#     # @staticmethod
#     # def _fit_OLD(event_counts, like_func, trans_period , penalty = None ):
#     #     """
        
#     #     """
#     #     n_days = len(event_counts)
#     #     cond_probs = Infect_Date_Estimator._make_cond_prob_mat(like_func, n_days)
#     #     d2=Infect_Date_Estimator._make_d2_mat(n_days)
#     #     if penalty is None:
#     #         pass
#     #     else:
#     #         ## Get initial guess
#     #         A = cond_probs.copy()
#     #         b= event_counts/event_counts.sum()
#     #         x = cp.Variable(A.shape[1])
#     #         constraints = []
#     #         constraints.append(x >= 0) 
#     #         constraints.append(x <= 1)
#     #         constraints.append(cp.sum(x) == 1)

#     #         objective = cp.Minimize(cp.sum_squares(A@x-b) + 1.*cp.sum_squares(d2@x) )
#     #         problem = cp.Problem(objective, constraints)
#     #         loss = problem.solve()
#     #         soln = x.value.copy()
            
#     #         ## get full solution with scipy optimize
#     #         A = cond_probs.copy()
#     #         b= event_counts/event_counts.sum()
#     #         M1 ,M2 = Infect_Date_Estimator._make_log_penalty_mats(n_days, window_size=trans_period[1]-1, like_func = like_func)
#     #         D = Infect_Date_Estimator._make_d1_mat( n_days-(trans_period[1] -1) )
            
#     #         def obj_func(x, A, b, M1, M2, D, penalty):
#     #             obj = np.sum( (A@x-b)**2) + \
#     #                     penalty*np.sum(( D@( np.log(np.abs(M1@x+0.005)) - np.log(np.abs(M2@x+0.005)) )
#     #                                     )**2 )
#     #             return obj
            
#     #         def jac_func(x,A, b, M1, M2, D, penalty):
#     #             M1_x = M1@x
#     #             M2_x = M2@x
#     #             jac_ls = -2.*b@A + 2.*x@np.transpose(A)@A
#     #             jac_pen = ( np.log(np.abs(M1_x + 0.005)) - np.log(np.abs(M2_x + 0.005))
#     #                                       )@(np.transpose(D)@D
#     #                                         )@( np.diag(np.sign(M1_x+0.005)/np.abs(M1_x+0.005))@M1 - \
#     #                                                    np.diag(np.sign(M2_x+0.005)/np.abs(M2_x+0.005))@M2 )
                
#     #             return jac_ls + 2.*penalty*jac_pen
                                               
#     #         opt_res =  sp.optimize.minimize( fun = obj_func, x0 = soln, 
#     #                                         jac = jac_func , 
#     #                                         args = (A, b, M1, M2, D, penalty),
#     #                                         method = 'trust-constr',
#     #                                        bounds = [(0.,1.)]*len(soln) ,
#     #                                        constraints = sp.optimize.LinearConstraint( np.ones( (1,len(soln)), dtype = float) ,
#     #                                                                                    lb = 1., ub = 1. )
#     #                                        )
#     #         least_sq_loss = np.sum( (A@opt_res.x - b)**2)
#     #         if not opt_res.success:
#     #             raise ValueError("sp.optimize.minimize failed to converge with message:\n{}".format(opt_res.message) )
#     #     return opt_res.x, opt_res.fun, least_sq_loss
    
#     @staticmethod
#     def _resample_events( event_series, n = 20 ):
#         day_index = np.array( [ (x-event_series.index[0]).days for x in event_series.index ] )
#         sample_size = int(event_series.sum())
#         resampled = np.random.choice(day_index,
#                                     size= n*sample_size ,
#                                     p = event_series.values / event_series.values.sum()  
#                                     )
#         resampled_df = pd.DataFrame( index = event_series.index, 
#                                     columns = list(range(n)),
#                                 data = 0.)
#         for iter_idx in range(n):
#             resampled_tmp = resampled[iter_idx*sample_size: (iter_idx+1)*sample_size]
#             resampled_df.iloc[:,iter_idx] =np.array([ np.sum(resampled_tmp == x) for x in day_index ])
#         return  resampled_df

#     @staticmethod
#     def _run_bootstram(event_series, penalties_test, like_func, trans_period, n_boot = 20):
#         like_func_Nzero = np.argmax(like_func > 0.)
#         n_days = len(event_series)
#         p_i_index = event_series.index.shift(-1*like_func_Nzero, freq='D')  ### shift index for infecation dates
#         p_i_from_boot = pd.DataFrame( index = p_i_index, 
#                                     columns = pd.MultiIndex.from_product( [ ["{:.3e}".format(p) for p in penalties_test ], list(range(n_boot)) ] ) ,
#                                     data = 0., 
#                                     dtype = float)
#         losses_from_boot = pd.Series(index = pd.MultiIndex.from_product( [ ["{:.3e}".format(p) for p in penalties_test ], list(range(n_boot)) ] ) ,
#                                     data = 0., 
#                                     dtype = float) 
#         ## Fit to bootstrap samples
#         events_resampled =  Infect_Date_Estimator._resample_events(event_series.copy(), n = n_boot)
#         with progressbar.ProgressBar(max_value=len(penalties_test)*n_boot) as bar:
#             bar_idx = 0
#             for p in penalties_test:
#                 for boot_idx in range(n_boot):
#                     p_i_given_obsCas, loss ,least_sq_loss = Infect_Date_Estimator._fit( event_counts =  events_resampled.iloc[:, boot_idx].values.copy(), 
#                                                                                         like_func = like_func,
#                                                                                         penalty = p,
#                                                                                         trans_period= trans_period )
#                     p_i_given_obsCas = pd.Series( index= p_i_index,  data = p_i_given_obsCas)
#                     _, p_i_full ,_ = Infect_Date_Estimator._rescale_p_i(p_i_given_obsCas, like_func, n_days)
#                     p_i_from_boot.loc[: , ("{:.3e}".format(p), boot_idx)] = p_i_full 
#                     losses_from_boot.loc[("{:.3e}".format(p), boot_idx)] = least_sq_loss 

#                     bar.update(bar_idx)
#                     bar_idx +=1

#         return p_i_from_boot, losses_from_boot

#     @staticmethod
#     def _elbow_analysis(fit_error, fit_var, penalties):
#         """
#         fit_error, fit_var, penalties - 1darrays 
#         """
#         sort_idxs = np.argsort(penalties)
#         fit_error = np.copy(fit_error[ sort_idxs ] )
#         fit_var = np.copy(fit_var[ sort_idxs ] )
#         penalties = np.copy(penalties[ sort_idxs ] )
        
#         d_error_d_var = np.diff(fit_error)/np.diff(fit_var)
        
#         means_init = np.array([ [d_error_d_var[0]] , [np.mean(d_error_d_var)]  ])
#         precisions_init =  np.array( [  [[1./(np.var(d_error_d_var)*0.5) ]], [[1./(np.var(d_error_d_var)*5.) ]] ] )
#         gmm = GaussianMixture(n_components=2, covariance_type='full', 
#                             means_init=means_init , 
#                             precisions_init = precisions_init ,
#                             ).fit(  d_error_d_var.reshape(-1,1) )
#         gmm_preds = gmm.predict(d_error_d_var.reshape(-1,1) )
#         penalty_best = float(penalties[1 + np.argmax(np.diff(gmm_preds) != 0)  ]  )
#         return penalty_best

#     def choose_penalty(self, penalties_test, method = "bootstrap",  **kwargs) :
#         """[summary]
        
#         Arguments:
#             penalties_test {iterable} -- iterable of floats 
        
#         Keyword Arguments:
#             n_boot {int} -- [description] (default: {20})
        
#         Returns:
#             [type] -- [description]
#         """
#         if method.lower()  == "bootstrap":
#             p_i_from_boot, losses_from_boot = self._run_bootstram(self.event_series.copy(), 
#                                                                     penalties_test, 
#                                                                     self.like_func.copy(), 
#                                                                     self.trans_period, **kwargs)
#             self.p_i_from_boot = p_i_from_boot.groupby(axis = 1, level = 0
#                                         ).apply( lambda x: pd.concat([x.mean(axis = 1), x.var(axis = 1)], keys=["mean", "var"]) 
#                                             ).unstack( level = 0)
#             objective_vs_penalty = losses_from_boot.groupby(level = 0).mean()
#             objective_vs_penalty.name = "mean_objective"
#             self.objective_vs_penalty =  objective_vs_penalty.to_frame()
#             self.objective_vs_penalty["var"] =  self.p_i_from_boot.loc[: , (slice(None) , "var") ].sum(axis = 0).droplevel(1)

#             self.penalty = self._elbow_analysis(np.sqrt(self.objective_vs_penalty["mean_objective"]).values , 
#                                                  np.sqrt(self.objective_vs_penalty["var"]).values , 
#                                                 self.objective_vs_penalty.index.values.astype(float) )
        
#         return self.penalty 

#     def plot_penalty_choice(self, ax = None):
#         if self.objective_vs_penalty is None:
#             raise Exception("Need to run choose penalty method first")
#         if ax is None:
#             fig, ax = plt.subplots(figsize = (12,4))
#         plot_data = self.objective_vs_penalty.copy()
#         plot_data.index = plot_data.index.values.astype(float) 
#         objective_best, var_best =  plot_data.loc[self.penalty, ["mean_objective" , "var"] ]


#         norm = colors.LogNorm(vmin =  np.min(plot_data.index.values), vmax = np.max(plot_data.index.values) )
#         ax.scatter( [ var_best,],  [objective_best,] , c = self.penalty, s = 200, norm = norm,
#                         edgecolor = "r", linewidth = 5, label = "best")
#         path = ax.scatter( plot_data["var"].values , 
#                            plot_data["mean_objective"].values , 
#                             c =plot_data.index.values , s= 100, norm = norm)
#         ax.legend(loc =1)
#         ax.set_ylabel("Square Error")
#         ax.set_xlabel("Total Variance (from bootstrap)")
#         fig = ax.get_figure()
#         fig.colorbar(path, ax = ax)
#         return ax

#     def est_transmission_rate(self, min_infected_frac = 0.0001 ):
#         """
#         infected_counts  - series with values — proportional to number of infected people and index — dates
#         trans_period  - 2-tuple representing left end open right end closed interval
#         """
#         infected_counts = self.p_i.copy()
#         est_start_idx = max( self.trans_period[1] - 1, np.argmax( infected_counts.values >=min_infected_frac) )
#         self.transmission_rates = pd.Series(index = infected_counts.index[est_start_idx:].copy() , data = 0. )

#         for idx in range( est_start_idx, len(infected_counts) ):
#             self.transmission_rates.iloc[idx - est_start_idx] = infected_counts.iloc[idx] / ( 
#                                                             infected_counts.iloc[idx-self.trans_period[1]+1 : idx-self.trans_period[0]+1].sum())
#         return self.transmission_rates.copy()
 
#     @staticmethod
#     def _make_d2_mat(size):
#         """
#         size - int — the number of rows and columns in matrix
#         """
#         d2 = np.zeros((size,size), dtype = float)
#         for i in range(size):
#             if i+2 < size:
#                  d2[i,i:i+3] = [1.,-2.,1.]
#         return d2
#     @staticmethod 
#     def _make_d1_mat(size):
#         d1 = np.zeros((size-1,size), dtype = float)
#         for i in range(0,size-1):
#             d1[i,i:i+2] = [-1., 1.]
#         return d1

#     def est_total_infect_by_day(self, age_group_data):
#         """
#         """
#         self.infect_count_all =  self.infect_count_cas / (age_group_data["mortality_rate"]*age_group_data["pop_frac"]).sum()
#         return self.infect_count_all.copy() 
    