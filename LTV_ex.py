#1. Libraries
import pandas                     as pd
import numpy                      as np
import matplotlib.patches         as mpatches
from scipy                        import stats
from bisect                       import bisect_left
from matplotlib                   import pyplot as plt
from statsmodels.othermod.betareg import BetaModel
#2. Functions
##2.1 Statistical tests class
class StatTest(object):
 #Calculate  statistic and P-value Chow test
 def chow(self, res1, res2, k, l):
  stat = (np.sum(res1**2)-np.sum(res2**2))*(len(res1)-l*k)/k/np.sum(res2**2)
  Pval = 1-stats.f.cdf(stat, k, len(res1)-l*k)
  return {'T-test': stat, 'P-value': Pval}
##2.2 Logistic function
def logistic(x):
 return 1/(1+np.exp(-x))
##2.3 Logodds function
def LogOdd(x):
 return np.log(x/(1-x))
# 3. Simulation
if __name__=='__main__':
 ##3.1 Simulation parameters
 np.random.seed(1)
 N          = 10000
 Int        = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
 Val        = ['A', 'B', 'C', 'D', 'E', 'F']
 Params     = pd.DataFrame({'Intercept': [LogOdd(0.015),  -4.08, -4.02,  -4.01, -4.10, LogOdd(0.99)],
                            'LTV':       [0            ,  0.13, 0.13,  0.36, 0.36, 0           ],
                            'precision': [np.nan]+[22]*4+[np.nan]},
                            index=['Params Group A','Params Group B','Params Group C','Params Group D','Params Group E','Params Group F'])
 ##3.2 Generate sample
 df         = pd.DataFrame({'ID'       : range(N),
                            'Group'    : map(lambda x: Val[bisect_left(Int, x)], np.random.uniform(0,1,N)),
                            'LTV'      : np.random.gamma(3,2,N),
                            'Intercept': np.ones(N),
                            'Res1'     : [0]*N,
                            'Res2'     : [0]*N,
                            'Res3'     : [0]*N })
 ##3.3 Calculate observed loss
 mu         = np.array([logistic(0.13*df.LTV-4.08), logistic(0.13*df.LTV-4.02), logistic(0.36*df.LTV-4.01), logistic(0.36*df.LTV-4.10)]).T
 phi        = 22
 alpha      = np.array([ mu[:,x] * phi         for x in range(4) ]).T
 beta       = np.array([ phi * ( 1 - mu[:,x] ) for x in range(4) ]).T
 df['Loss'] = df.apply(lambda x: np.random.uniform(0   , 0.03)               if x.Group=='A' else
                                 np.random.uniform(0.98, 1   )               if x.Group=='F' else
                                 np.random.beta(alpha[x.ID,0], beta[x.ID,0]) if x.Group=='B' else
                                 np.random.beta(alpha[x.ID,1], beta[x.ID,1]) if x.Group=='C' else
                                 np.random.beta(alpha[x.ID,2], beta[x.ID,2]) if x.Group=='D' else
                                 np.random.beta(alpha[x.ID,3], beta[x.ID,3]), axis=1)
 df.Loss             = df.Loss.apply(lambda x: min(0.999999, x))
 ##3.4 Plot relationship between LTV and observed Loss
 df.plot.scatter(x='LTV', y='Loss')
 plt.show()
 colors = {'A': 'orange', 'B': 'purple', 'C': 'blue', 'D': 'red', 'E': 'brown', 'F': 'yellow'}
 Clist  = [colors[x] for x in df.Group]
 df.plot.scatter(x='LTV', y='Loss', c=Clist)
 handles = [mpatches.Patch(color=colour, label=label) for label, colour in [('A','orange'), ('B','purple'), ('C', 'blue'), ('D','red'), ('E', 'brown'), ('F', 'yellow')]]
 plt.legend(handles=handles, loc='lower right', frameon=True)
 plt.show()
 ##3.5 Beta regressions
 ###3.5.1 Beta regression with dummy variables
 sample     = pd.get_dummies(df,columns=['Group'], drop_first=True)
 LinM       = "Loss ~ LTV + Group_B + Group_C + Group_D + Group_E + Group_F"
 M          = BetaModel.from_formula(LinM, sample)
 Results    = M.fit()
 df['Res3'] = df.Loss - logistic(Results.params[:-1] @ sample[['Intercept', 'LTV', 'Group_B', 'Group_C', 'Group_D', 'Group_E', 'Group_F']].T)
 print(Results.summary())
 ###3.5.2 Beta regression with only LTV
 LinM       = "Loss ~ LTV"
 M          = BetaModel.from_formula(LinM, df)
 Results    = M.fit()
 df['Res1'] = df.Loss - logistic(Results.params[:-1] @ df[['Intercept', 'LTV']].T)
 Params     = pd.concat([Params, pd.DataFrame(Results.params, columns=['Estimate all observations']).T])
 ###3.5.3 Beta regression with only LTV per collateral cohort
 for x in Val:
  ix                 = x==df.Group
  sample             = df[ix]
  M                  = BetaModel.from_formula(LinM, sample)
  Results            = M.fit()
  df.loc[ix, 'Res2'] = sample.Loss - logistic(Results.params[:-1] @ sample[['Intercept', 'LTV']].T)
  Params             = pd.concat([Params, pd.DataFrame(Results.params, columns=['Estimate Group '+x]).T])
 ###3.5.4 Plot different estimates
 df[['Res1','Res2','Res3']].rename(columns={'Res3':'Regression with dummy variables','Res1':'Regression only LTV','Res2':'Regression only LTV per collateral cohort'}).plot.kde()
 plt.xlim(-1, 1)
 plt.show()
 ##3.6 Print regression estimates
 print(Params)
 ##3.7 Chow test
 print(StatTest().chow(df.Res1, df.Res2, 3, len(df.Group.unique())))
