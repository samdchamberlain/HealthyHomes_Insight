# required packages
import numpy as np
import pandas as pd

# general Shannon entropy functions for joint and marginal entropies
def entropy(x, bins = 10):
    
    counts = x.value_counts(bins = bins)
    probs = counts / np.sum(counts)
    
    return - np.sum(probs * np.log2(probs))

def joint_entropy(x, y, bins = 10):
    
    x.reset_index(drop = True, inplace = True)
    y.reset_index(drop = True, inplace = True)
    combined = pd.concat([x, y], axis=1).dropna() #drop all rows with NA in both variables
    j_probs = pd.crosstab(pd.cut(combined.iloc[:, 0], bins = bins), 
                          pd.cut(combined.iloc[:, 1], bins = bins)) / len(combined)
    
    return - np.sum(np.sum(j_probs * np.log2(j_probs)))

def entropy_3d(x, y, z, bins = 10):
    
    x.reset_index(drop = True, inplace = True)
    y.reset_index(drop = True, inplace = True)
    z.reset_index(drop = True, inplace = True)
    combined = pd.concat([x, y, z], axis = 1).dropna()
    
    j_probs = pd.crosstab(pd.cut(combined.iloc[:, 0], bins = bins),
                          [pd.cut(combined.iloc[:, 1], bins = bins),
                           pd.cut(combined.iloc[:, 2], bins = bins)]) / len(combined)
    
    return - np.sum(np.sum(j_probs * np.log2(j_probs)))

# Mutual information calculator
def mutual_information(x, y, bins = 10, normalize = True):
    
    Hx = entropy(x, bins = bins)
    Hy = entropy(y, bins = bins)
    Hxy = joint_entropy(x, y, bins = bins)
    
    if normalize == True:
        MI = (Hx + Hy - Hxy) / Hy
    else:
        MI = Hx + Hy - Hxy
        
    return MI

def transfer_entropy(x, y, xlag, ylag = 1, bins = 10, normalize = True):
    
    #lag x and y variables
    x_lagged = x.shift(xlag).rename('x_lagged')
    y_lagged = y.shift(ylag).rename('y_lagged')
    
    #calculate joint and marginal entropies for transfer entropy calculation
    Hxy = joint_entropy(x_lagged, y_lagged, bins = bins)
    Hyy = joint_entropy(y, y_lagged, bins = bins)
    Hyl = entropy(y_lagged, bins = bins)
    Hxyz = entropy_3d(x_lagged, y_lagged, y, bins = bins)
    Hy = entropy(y, bins = bins)
    
    if normalize == True:
        TE = (Hxy + Hyy - Hyl - Hxyz) / Hy
    else:
        TE = Hxy + Hyy - Hyl - Hxyz
    
    return TE 

def lag_TE(x, y, lags, bins = 10, normalize = True):
    
    lagged_out = [transfer_entropy(x, y, xlag = (i + 1), bins=bins, ylag=1, normalize=normalize) for i in range(lags)]
    return pd.Series(lagged_out)

# Calculate a mutual information timeseries in monthly chunks
# includes loop for calculating shuffled surrogates    
def MI_timeseries(x_var, y_var, bins = 10, normalize = True,
                  runs = 100, alpha = 0.05, MC_runs = True):
    
    #combine variables and create monthly stamp
    combined = pd.concat([x_var, y_var], axis=1).dropna()
    combined['Yr_mnth'] = combined.index.strftime('%Y%m')
    cols = combined.columns
    
    #subset data my month, calculate and save mutual information
    time_index = combined['Yr_mnth'].unique()
    MI_out = np.empty(len(time_index)) #empty MI array
    
    if MC_runs == True:
        MI_MC = np.empty([len(time_index), runs]) #raw MC runs matrix
    
    for i in range(len(time_index)):
        
        monthly = combined[combined.Yr_mnth == time_index[i]]
        MI_out[i] = mutual_information(monthly[cols[0]], monthly[cols[1]], 
                                       bins = bins, normalize=normalize)
        
        if MC_runs == True:
            
            for j in range(runs):
                x_shuffle = monthly[cols[0]].sample(len(monthly), replace = False) #shuffle x
                
                MI_MC[i, j] = mutual_information(x_shuffle, monthly[cols[1]], 
                                                 bins = bins, normalize = normalize)         
    
    if MC_runs == True:
        MI_MC = np.percentile(MI_MC, q = (1 - alpha) * 100, axis = 1)                
        return MI_out, MI_MC
    
    return MI_out

def TE_timeseries(x_var, y_var, lags, bins=10, normalize = True,
                  runs = 10, alpha = 0.05, MC_runs = True):
    
    #remove NAs and subset growing season only data (DOY 100-300)
    combined = pd.concat([x_var, y_var], axis = 1).dropna()
    combined = combined[(combined.index.dayofyear > 100) & (combined.index.dayofyear < 300)]
    cols = combined.columns
    
    #iterate across each years growing season
    year = combined.index.year.unique()
    TE_out = np.empty([lags, len(year)])
    
    if MC_runs == True:
        TE_MC = np.empty([lags, len(year), runs])
    
    for i in range(len(year)):
        
        subset = combined[combined.index.year == year[i]]
        TE_out[:, i] = lag_TE(subset[cols[0]], subset[cols[1]], lags=lags, bins=bins, normalize=normalize)
        
        if MC_runs == True:
            
            for j in range(runs):
                
                x_shuffle = subset[cols[0]].sample(len(subset), replace = False) #shuffle x variable
                TE_MC[:, i, j] = lag_TE(x_shuffle, subset[cols[1]], lags=lags, bins=bins, normalize=normalize)
    
    if MC_runs == True:
        TE_MC = pd.DataFrame(np.percentile(TE_MC, q = (1 - alpha) * 100, axis = 2), 
                             columns=combined.index.year.unique())
        
    TE_out = pd.DataFrame(TE_out, columns=combined.index.year.unique())
    TE_out.index = TE_out.index + 1
    
    return TE_out, TE_MC