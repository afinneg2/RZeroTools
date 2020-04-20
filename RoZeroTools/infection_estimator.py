
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt  
import sys
import scipy as sp
import cvxpy as cp
from datetime import timedelta




class Infect_Date_Estimator(object):
    def __init__(self, event_series, like_func, trans_period = (1,8)  ):
        """
        event_series -  index is of type datetime64[ns]
        like_func - pmf for likelihood of event (death/hospitalization) as function of days after infection
        """

        self.event_series = self._preprocess_event_series(event_series, like_func)
        self.date_earliest = self.event_series.index[0]
        self.date_latest = self.event_series.index[-1]
        self.n_days = (self.date_latest - self.date_earliest).days + 1
        
        self.like_func = like_func
        self._like_func_Nzero = np.argmax(like_func > 0.)
        self.N_obsCas = self.event_series.sum()
        if trans_period[0] != 1:
            raise ValueError("Code only supports models with transmission period beginning 1 d after infection")
        self.trans_period = trans_period
        
        self.infect_frac_cas = None
        self.infect_count_cas = None
        self.infect_count_all = None

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

    @property
    def conditional_prob_mat(self,):
        cond_probs = self._make_cond_prob_mat(self.like_func, self.n_days, given_obsCas=True)
        #cond_probs =  pd.DataFrame( data= cond_probs, index =  "" , columns = "" ) 
        return cond_probs
    @property
    def log_penalty_mats(self):
        M1 , M2= Infect_Date_Estimator._make_log_penalty_mats(self.n_days, window_size=self.trans_period[1]-1, like_func = self.like_func)
        return M1, M2
    @property 
    def D1(self):
        D1 = Infect_Date_Estimator._make_d1_mat( self.n_days-(self.trans_period[1] -1) )
        return D1

    def est_casualty_infect_dates(self, d_preceeding = None, smooth_penalty =None  ): 

        p_i_given_obsCas, loss  = self._fit( event_counts = self.event_series.values.copy(), 
                                            like_func = self.like_func,
                                            penalty = smooth_penalty,
                                            trans_period= self.trans_period)
        
        ## shift index
        p_i_index = self.event_series.index.shift(-1*self._like_func_Nzero, freq='D')
       
        self.p_i_given_obsCas = pd.Series( index= p_i_index,  data = p_i_given_obsCas)
        self.N_i_obsCas_expected =  self.p_i_given_obsCas*self.N_obsCas
        
        self.p_i, self._p_i_full, self._frac_expected = self._rescale_p_i(self.p_i_given_obsCas, self.like_func, self.n_days)
        self._N_cas_full = self.N_obsCas/(1. - self._frac_expected) 
        self.N_cas = self._N_cas_full*(self._p_i_full.loc[self.p_i.index].sum())
        
        # self.p_e = self._calc_p_e(self.p_i.copy(), self.like_func.copy() )
        # self.N_i_cas_expected = self.p_e * self.N_cas
        return self.p_i_given_obsCas, self.p_i, #self.N_i_cas_expected
     
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
    def _fit(event_counts, like_func, trans_period , penalty = None ):
        """
        
        """
        n_days = len(event_counts)
        cond_probs = Infect_Date_Estimator._make_cond_prob_mat(like_func, n_days)
        d2=Infect_Date_Estimator._make_d2_mat(n_days)
        if penalty is None:
            pass
        else:
            ## Get initial guess
            A = cond_probs.copy()
            b= event_counts/event_counts.sum()
            x = cp.Variable(A.shape[1])
            constraints = []
            constraints.append(x >= 0) 
            constraints.append(x <= 1)
            constraints.append(cp.sum(x) == 1)

            objective = cp.Minimize(cp.sum_squares(A@x-b) + 1.*cp.sum_squares(d2@x) )
            problem = cp.Problem(objective, constraints)
            loss = problem.solve()
            soln = x.value.copy()
            
            ## get full solution with scipy optimize
            A = cond_probs.copy()
            b= event_counts/event_counts.sum()
            M1 ,M2 = Infect_Date_Estimator._make_log_penalty_mats(n_days, window_size=trans_period[1]-1, like_func = like_func)
            D = Infect_Date_Estimator._make_d1_mat( n_days-(trans_period[1] -1) )
            
            def obj_func(x, A, b, M1, M2, D, penalty):
                obj = np.sum( (A@x-b)**2) + \
                        penalty*np.sum(( D@( np.log(np.abs(M1@x+0.005)) - np.log(np.abs(M2@x+0.005)) )
                                        )**2 )
                return obj
            
            def jac_func(x,A, b, M1, M2, D, penalty):
                M1_x = M1@x
                M2_x = M2@x
                jac_ls = -2.*b@A + 2.*x@np.transpose(A)@A
                jac_pen = ( np.log(np.abs(M1_x + 0.005)) - np.log(np.abs(M2_x + 0.005))
                                          )@(np.transpose(D)@D
                                            )@( np.diag(np.sign(M1_x+0.005)/np.abs(M1_x+0.005))@M1 - \
                                                       np.diag(np.sign(M2_x+0.005)/np.abs(M2_x+0.005))@M2 )
                
                return jac_ls + 2.*penalty*jac_pen
                                               
            opt_res =  sp.optimize.minimize( fun = obj_func, x0 = soln, 
                                            jac = jac_func , 
                                            args = (A, b, M1, M2, D, penalty),
                                            method = 'trust-constr',
                                           bounds = [(0.,1.)]*len(soln) ,
                                           constraints = sp.optimize.LinearConstraint( np.ones( (1,len(soln)), dtype = float) ,
                                                                                       lb = 1., ub = 1. )
                                           )
            print(opt_res.success)
            print(opt_res.message)
        return opt_res.x, opt_res.fun
    
    def est_transmission_rate(self, min_infected_frac = 0.0001 ):
        """
        infected_counts  - series with values — proportional to number of infected people and index — dates
        trans_period  - 2-tuple representing left end open right end closed interval
        """
        infected_counts = self.p_i.copy()
        est_start_idx = max( self.trans_period[1] - 1, np.argmax( infected_counts.values >=min_infected_frac) )
        self.transmission_rates = pd.Series(index = infected_counts.index[est_start_idx:].copy() , data = 0. )

        for idx in range( est_start_idx, len(infected_counts) ):
            self.transmission_rates.iloc[idx - est_start_idx] = infected_counts.iloc[idx] / ( 
                                                            infected_counts.iloc[idx-self.trans_period[1]+1 : idx-self.trans_period[0]+1].sum())
        return self.transmission_rates.copy()

    
    @staticmethod
    def _rescale_p_i(p_i_given_obsCas, like_func, n_days, ):
        """
        Calculates probabilites of infection on certain day from probabilies of infection on that day condition on observed death.
        Also Calculates the fraction of infections that have occurred so far for which casualties are expected.
        """
        like_func = like_func[np.argmax(like_func>0.) :].copy()
        p_i_full = pd.Series(index = p_i_given_obsCas.index, 
                        data = [ x/np.sum(like_func[:n_days-d]) for d,x in enumerate(p_i_given_obsCas.values) ])
        frac_expected = 1. - 1./p_i_full.sum()
        p_i_full = p_i_full/p_i_full.sum()
        
        d_truncate = np.argmax(np.cumsum(like_func) > 0.1)
        print(d_truncate)
        p_i =  p_i_full.iloc[:-1*d_truncate]/ p_i_full.iloc[:-1*d_truncate].sum()
        return p_i, p_i_full, frac_expected
 
    @staticmethod
    def _calc_p_e(p_i, like_func,):
        cond_probs = Infect_Date_Estimator._make_cond_prob_mat(like_func, len(p_i), given_obsCas = False)
        p_e = np.dot(cond_probs , p_i.values[:,None]).squeeze()
        
        p_e = pd.Series(data = p_e)
        p_e_index = p_i.index.copy()
        p_e_index = p_e_index.append( p_e_index[-1] + pd.timedelta_range( start = timedelta(days = 1), 
                                                                        end = timedelta(days = len(like_func) -1), 
                                                                        freq = 'd'  ) 
                                     )
        p_e.index= p_e_index 
        return p_e

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
    def _make_d1_mat(size):
        d1 = np.zeros((size-1,size), dtype = float)
        for i in range(0,size-1):
            d1[i,i:i+2] = [-1., 1.]
        return d1

    def est_total_infect_by_day(self, age_group_data):
        """
        """
        self.infect_count_all =  self.infect_count_cas / (age_group_data["mortality_rate"]*age_group_data["pop_frac"]).sum()
        return self.infect_count_all.copy() 
    