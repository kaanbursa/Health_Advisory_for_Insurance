import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

def drop_cols(df,n):
    """
    Drop column if number of null values is higher than n
    """
    for col in df.columns:
    
        nulvals = sum(df[col].isna())
        if nulvals > n:
            df.drop(col,axis=1,inplace=True)
            
    return df


def drop_ci_quartile_col(df):
    for col in df.columns:
        if '95% CI - Low' in df[col].tolist() or '95% CI - High' in df[col].tolist() or 'Quartile' in df[col].tolist():
            df.drop(col,inplace=True,axis=1)
    return df
            
            
def rename_cols(columns):
    renamed = map(lambda x: x.replace('# ','').replace('% ', '').replace(' ','_').replace('-','_'), list(columns))
    return renamed


def corr_matrix(df):
    corr = df.corr()
    # Mask the upper half
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(11, 9))

    ax.set_title('Correlation Matrix of Life Expectancy by County')
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    
def iterative_feature_selector(df):
    it_df = df.copy()
    for i in range(it_df.shape[1]):
        ind_vars = " + ".join(it_df.drop('Life_Expectancy',axis=1).columns)

        formula = "Life_Expectancy ~ " + ind_vars
        model = smf.ols(formula=formula,data=it_df).fit()
        pvals = {model.params.index[j]:model.pvalues[j] for j in range(len(model.params))}
        aic = model.aic
        dct = {k:v for (k,v) in pvals.items() if v > 0.05}
        try:
            key_max = max(dct.keys(), key=(lambda k: dct[k]))
        except:
            continue
        if bool(dct):
            it_df.drop(key_max,axis=1,inplace=True)
    return it_df