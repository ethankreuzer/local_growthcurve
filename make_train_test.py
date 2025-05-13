#!/usr/bin/env python
# coding: utf-8

# In[1]:
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import datamol as dm
from sklearn.metrics import roc_auc_score

# # Data processing 

# The GrowthCurve.pkl dataframe is not the format that we want to be fed to a model. Needs to be reformatted so every OD values is its own row.
# 
# Notes about dataset, every Active compound has all the time points.

# In[2]:

def one_hot_time_encoding(time_series):
    """One-hot encode the time feature."""
    return pd.get_dummies(time_series, prefix='time').astype(int)

def one_hot_concentration_encoding(conc_series):
    """One-hot encode the concentration feature."""
    return pd.get_dummies(conc_series, prefix='conc').astype(int)

def reformat_growth_curve_data(df, time_encoder, conc_encoder):
    """
    Reformats the growth curve dataframe so that each OD measurement is its own sample.
    
    Parameters:
        df (pd.DataFrame): The original dataframe.
        time_encoder (callable): A function that takes a pd.Series of times and returns its encoded dataframe.
        conc_encoder (callable): A function that takes a pd.Series of concentrations and returns its encoded dataframe.
    
    Returns:
        pd.DataFrame: The reformatted dataframe.
    """
    # Identify all columns that represent timepoints (assuming they start with 't_')
    time_cols = [col for col in df.columns if col.startswith('t_')]

    
    # Melt the dataframe to convert wide-format timepoint columns into a single 'time' column.
    df_long = pd.melt(
        df,
        id_vars=['Compound', 'Concentration', 'Activity', 'Smiles'],
        value_vars=time_cols,
        var_name='time',
        value_name='OD',
    )
    
    # Convert the time column to numeric values by stripping the 't_' prefix.
    df_long['time'] = df_long['time'].str.lstrip('t_').astype(float)

    

    conc_map = {0.2: 0, 1.2: 1, 3.13: 2, 7.9: 3, 12.5: 4, 50.0: 5}
    time_map = {0.0: 0, 2.08: 1, 4.16: 2, 6.24: 3, 8.32: 4, 10.4: 5, 12.48: 6}
    
    # Add new index columns using the mappings.
    df_long['indx_conc'] = df_long['Concentration'].map(conc_map)
    df_long['indx_time'] = df_long['time'].map(time_map)

    
    # Encode the time and concentration using the provided encoder functions.
    time_encoded = time_encoder(df_long['time'])
    conc_encoded = conc_encoder(df_long['Concentration'])
    
    # Optionally drop the original time and concentration columns and add the new encoded columns.
    df_encoded = pd.concat(
        [df_long.drop(columns=['time', 'Concentration']), time_encoded, conc_encoded],
        axis=1
    )
    
    return df_encoded

def plot_activity_ratio_heatmap(df):
    """
    Creates a single heatmap showing, for each combination of concentration and timepoint,
    the ratio of active compounds among the total compounds, with each cell annotated as "actives/total".
    
    The DataFrame df is assumed to contain:
        - 'indx_conc': integer index for concentration
        - 'indx_time': integer index for timepoint
        - 'Activity': with values 'Active' or 'Inactive'
    
    The concentration mapping is:
        0 -> 0.2, 1 -> 1.2, 2 -> 3.13, 3 -> 7.9, 4 -> 12.5, 5 -> 50.0
    
    The time mapping is:
        0 -> 0.0, 1 -> 2.08, 2 -> 4.16, 3 -> 6.24, 4 -> 8.32, 5 -> 10.4, 6 -> 12.48
    """
    # Group by concentration and time for all compounds
    total_counts = df.groupby(['indx_conc', 'indx_time']).size().unstack(fill_value=0)
    
    # Group by concentration and time for active compounds only
    active_counts = df[df['Activity'] == 'Active'].groupby(['indx_conc', 'indx_time']).size().unstack(fill_value=0)
    
    # Reindex active_counts to ensure it has the same structure as total_counts.
    active_counts = active_counts.reindex_like(total_counts).fillna(0).astype(int)
    
    # Compute the fraction of active compounds; avoid division by zero by replacing zeros temporarily.
    fraction = active_counts / total_counts.replace(0, 1)
    fraction = fraction.fillna(0)
    
    # Create an annotation DataFrame with the format "active/total"
    annot = active_counts.astype(str) + "/" + total_counts.astype(str)
    
    # Define mapping dictionaries to translate index integers back to legible labels.
    conc_labels = {0: '0.2', 1: '1.2', 2: '3.13', 3: '7.9', 4: '12.5', 5: '50.0'}
    time_labels = {0: '0.0', 1: '2.08', 2: '4.16', 3: '6.24', 4: '8.32', 5: '10.4', 6: '12.48'}
    
    # Plot the heatmap using the fraction data, but annotate with "active/total" strings.
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(fraction, annot=annot, fmt="", cmap="viridis", cbar_kws={'label': 'Fraction Active'})
    
    # Set x-axis tick labels to legible time values and y-axis tick labels to legible concentration values.
    ax.set_xticklabels([time_labels.get(int(tick), tick) for tick in fraction.columns])
    ax.set_yticklabels([conc_labels.get(int(tick), tick) for tick in fraction.index], rotation=0)
    
    ax.set_xlabel('Timepoint')
    ax.set_ylabel('Concentration')
    ax.set_title('Active / Total Compounds Test set')
    
    plt.tight_layout()
    #plt.show()
    plt.savefig('/home/ethan/GrowthCurve/plots/test_active_counts_heatmap.png')

#Get the Fingerprints

def compute_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return None


def maccs_to_fp(smile):
    try:
        
        return dm.to_fp(smile, fp_type="maccs")
    except Exception as e:
       
        print(f"Error processing SMILES '{smile}': {e}")
        return np.nan
    

def ecfp_to_fp(smile):
    try:
        
        return dm.to_fp(smile, fp_type="ecfp")
    except Exception as e:
       
        print(f"Error processing SMILES '{smile}': {e}")
        return np.nan
    

def rdkit_to_fp(smile):
    try:
        
        return dm.to_fp(smile, fp_type="rdkit")
    except Exception as e:
       
        print(f"Error processing SMILES '{smile}': {e}")
        return np.nan

def train_test_split(df,percent):

    scaffold_to_indices = defaultdict(list)

    for idx, scaffold in enumerate(df["scaffold"]):
        scaffold_to_indices[scaffold].append(idx) #indx of samples with same scaffold

    scaffolds = list(scaffold_to_indices.keys())

    
    np.random.shuffle(scaffolds)

    split_idx = int(percent * len(scaffolds))
    train_scaffolds = scaffolds[:split_idx]
    test_scaffolds = scaffolds[split_idx:]

    train_indices = [idx for scaffold in train_scaffolds for idx in scaffold_to_indices[scaffold]]
    test_indices = [idx for scaffold in test_scaffolds for idx in scaffold_to_indices[scaffold]]

    df_train = df.iloc[train_indices]
    df_test = df.iloc[test_indices]
    
    return df_train, df_test

def plot_auroc_heatmap(df_test, X_test, y_test, model):
    """
    Plots a single heatmap showing the per-subset AUROC for each combination
    of timepoint and concentration index in df_test, using the trained 'model'.
    
    Each cell in the heatmap displays the AUROC value (0.00 - 1.00).
    Cells for which AUROC cannot be computed (e.g., only one class in that subset)
    are set to NaN.
    
    Parameters:
    -----------
    df_test : pd.DataFrame
        The test set containing 'indx_conc' and 'indx_time' columns (and 'Activity' if needed).
    X_test : pd.DataFrame
        The feature matrix for the test set (same index as df_test).
    y_test : np.array or pd.Series
        The binary labels (0 or 1) for the test set.
    model : trained classifier
        A scikit-learn classifier that supports `predict_proba`.
    """
    
    # Define reverse mappings for axis labels (adjust to your actual mapping if different).
    conc_labels = {0: '0.2', 1: '1.2', 2: '3.13', 3: '7.9', 4: '12.5', 5: '50.0'}
    time_labels = {0: '0.0', 1: '2.08', 2: '4.16', 3: '6.24', 4: '8.32', 5: '10.4', 6: '12.48'}
    
    # Get all unique concentration and time indices in the test set
    unique_concs = sorted(df_test['indx_conc'].unique())
    unique_times = sorted(df_test['indx_time'].unique())
    
    # Create an empty DataFrame to hold the AUROC values
    auroc_df = pd.DataFrame(index=unique_concs, columns=unique_times, dtype=float)
    
    # Iterate over each combination of concentration and time
    for c in unique_concs:
        for t in unique_times:

            # Filter df_test for rows with this concentration & time

            mask = (df_test['indx_conc'] == c) & (df_test['indx_time'] == t)
            
            subset_indices = df_test[mask].index
            
            # If there are fewer than 2 samples, AUROC isn't meaningful
            if len(subset_indices) < 2:
                auroc_df.loc[c, t] = np.nan
                print('NaN value inserted')
                continue
            
            # Extract labels and features for this subset
            y_subset = y_test[subset_indices]
            X_subset = X_test.loc[subset_indices]
            
            # If the subset has only one class (all 0 or all 1), AUROC is not defined
            if len(np.unique(y_subset)) < 2:
                auroc_df.loc[c, t] = np.nan
            else:
                # Compute predicted probabilities for the positive class
                y_pred_proba = model.predict_proba(X_subset)[:, 1]
                # Compute the AUROC
                auroc = roc_auc_score(y_subset, y_pred_proba)
                auroc_df.loc[c, t] = auroc
    
    # Plot the heatmap of AUROCs
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        auroc_df,
        annot=True,        # Show numeric AUROC values in each cell
        fmt=".2f",         # Format to two decimal places
        cmap="viridis",
        vmin=0, vmax=1,    # AUROC ranges from 0 to 1
        cbar_kws={"label": "AUROC"}
    )
    
    # Relabel the x-axis and y-axis ticks with human-readable labels
    ax.set_xticklabels([time_labels.get(t, str(t)) for t in unique_times])
    ax.set_yticklabels([conc_labels.get(c, str(c)) for c in unique_concs], rotation=0)
    
    ax.set_xlabel("Time Index (Mapped)")
    ax.set_ylabel("Concentration Index (Mapped)")
    ax.set_title("Per-Subset AUROC (Test Set)")
    
    plt.tight_layout()
    plt.savefig('/home/ethan/plots/RF_AUROC_heatmap')


raw_GrowthCurve = pd.read_pickle('/home/ethan/GrowthCurve/data/GrowthCurve.pkl')

# In[5]:

new_df = reformat_growth_curve_data(raw_GrowthCurve, one_hot_time_encoding, one_hot_concentration_encoding) #many NaNs as OD values here
df = new_df.dropna(subset=['OD'])
df = df.dropna(subset=['Smiles'])

# Once we remove all the NaN OD and Smiles values, we go from 314664 rows to 187720 (remove NaN OD) to 180944 (remove NaN Smiles)

# In[6]:

# # Add Fingerprints and Scaffolds

# In[ ]:



# Assume df has a "SMILES" column representing molecules


df["scaffold"] = df["Smiles"].apply(compute_scaffold)
df["maccs_fp"] = df["Smiles"].apply(maccs_to_fp)
df["ecfp_fp"] = df["Smiles"].apply(ecfp_to_fp)
df["rdkit_fp"] = df["Smiles"].apply(rdkit_to_fp)


df=df.dropna(subset=["maccs_fp", "ecfp_fp", "rdkit_fp"]) 

df.to_pickle("/home/ethan/GrowthCurve/data/GrowthCurve_processed.pkl")


# Now need to turn the fingerprint entries into individual features. There are no NaN values in the df at this point

# In[8]:


df=pd.read_pickle("/home/ethan/GrowthCurve/data/GrowthCurve_processed.pkl")

maccs_fp=pd.DataFrame(np.vstack(df['maccs_fp']))
ecfp_fp=pd.DataFrame(np.vstack(df['ecfp_fp']))
rdkit_fp=pd.DataFrame(np.vstack(df['rdkit_fp']))

fp_features = pd.concat([maccs_fp,ecfp_fp,rdkit_fp], axis=1)
fp_features.columns = [str(i) for i in range(fp_features.shape[1])]

concatenated_df = pd.concat([df.reset_index(drop=True), fp_features.reset_index(drop=True)], axis=1)

#The columns maccs_fp, ecfp_fp and rdkit_fp are still in this dataframe!

df_train, df_test = train_test_split(concatenated_df,0.8)

df_train.to_pickle('/home/ethan/GrowthCurve/data/df_train.pkl')
df_test.to_pickle('/home/ethan/GrowthCurve/data/df_test.pkl')


plot_activity_ratio_heatmap(df_test)