#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
import scipy.stats as stats

from matplotlib import style
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# This is the full Current Population Survey (CPS) dataset for the years 2000-2011. This is taken from monthly survey data that contains over 2.4 million rows for 103 variables, among which are several healthcare coverage variables (private insurance coverage, VA/military coverage, Medicaid coverage) as well as a health outcomes variable. Most of the other variables are demographic (age, sex, race, marital status, veteran status, citizenship status), while others relate to the labor market (income, industry, occupation, employment status/self-employment). Using these variables, we will try to explore the effect expanded insurance coverage under the 2006 Massachusetts Healthcare Reform on both health outcomes and self-employment.

# In[2]:


full_data = pd.read_csv("cps_00002.csv.gz", compression="gzip")
full_data


# Checking for null columns and rows

# In[3]:


full_data.columns[full_data.isna().any()].tolist()


# In[4]:


is_NaN = full_data.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = full_data[row_has_NaN]
rows_with_NaN[['YEAR', 'VERIFY', 'SCHIPLY']]


# For the year 2000, the variables 'VERIFY' (Verification: Did individual actually have health insurance) and 'SCHIPLY' (State Children's Health Insurance Program coverage last year) were null-valued. Note that health insurance data asks about patient's coverage in the previous year. Since these variables directly relate to insurance coverage, it would make more sense to drop the one year which doesn't have data on either of the variables rather than drop both of the variables entirely. So, for the purposes of this analysis, we will omit the year 2000 (health coverage for 1999). This still gives us a pre-treatment period from 2000 to 2005, which is still enough time to establish a pre-treatment trend with a post-treatment period of 2006 to 2010.

# In[5]:


full_data.dropna(axis=0, inplace=True)
full_data.reset_index(drop=True)


# In[6]:


data2001 = full_data.loc[full_data['YEAR'] == 2001]
data2001


# Looking at 2001 estimate of individuals nationwide covered by any Private Health Insurance (HCOVPRIV) - Excludes Medicare/Medicaid, Military, SCHIP, and other public health insurance plans - Subcategories: Employer-Sponsored, Individually-Purchased, Covered by someone outside the home
# 
# 2 - Covered by private insurance, 1 - Not covered by private insurance

# In[7]:


data2001['HCOVPRIV'].value_counts(normalize=True)


# Massachusetts private insurance coverage estimates 2001-2011

# In[8]:


for year in range(2001, 2012):
    data_year = full_data.loc[full_data['YEAR'] == year]
    print("Year:", year)
    print((data_year.loc[data_year['STATEFIP'] == 25])['HCOVPRIV'].value_counts(normalize=True), "\n")


# The 2006 reform also saw the expansion of Massachusetts' Medicaid program, but did not expand already-existing coverage for Medicare, SCHIP, or Military insurance programs.
# 
# Massachusetts Medicaid insurance coverage estimates 2001-2011
# 
# 1 - Not covered by Medicaid, 2 - Covered by Medicaid

# In[9]:


for year in range(2001, 2012):
    data_year = full_data.loc[full_data['YEAR'] == year]
    print("Year:", year)
    print((data_year.loc[data_year['STATEFIP'] == 25])['HIMCAIDLY'].value_counts(normalize=True), "\n")


# Now we want to see how coverage in Massachusetts compared to the rest of the US from 2001 to 2011. We will be excluding two states from this analysis, Hawaii and Oregon. Hawaii implemented a mandatory health insurance program in 1975, so their coverage rates would skew the data, which is meant to compare Massachusetts to states which DID NOT expand health care coverage through insurance mandates. Oregon conducted a Randomized Controlled Trial (RCT) where they expanded Medicaid coverage through an insurance lottery. Since this would bias the tail-end of the event study for Massachusetts, we will exclude Oregon from the data as well.

# In[10]:


control_states = full_data[(full_data.STATEFIP != 15) & (full_data.STATEFIP != 41) & (full_data.STATEFIP != 25)]
control_states.reset_index(drop=True)
control_states


# In[11]:


ma_data = full_data[full_data.STATEFIP == 25]
ma_data.reset_index(drop=True)
ma_data


# In[12]:


c = {'YEAR': [np.nan], 'priv_covered': [np.nan], 'priv_uncovered': [np.nan], 'priv_total': [np.nan], 'priv_prop_covered': [np.nan], 'priv_prop_uncovered': [np.nan], 'mcaid_covered': [np.nan], 'mcaid_uncovered': [np.nan], 'mcaid_total': [np.nan], 'mcaid_prop_covered': [np.nan], 'mcaid_prop_uncovered': [np.nan], 'total_uninsured': [np.nan], 'prop_uninsured': [np.nan]}
control_groups = pd.DataFrame(data=c)

m = {'YEAR': [np.nan], 'priv_covered': [np.nan], 'priv_uncovered': [np.nan], 'priv_total': [np.nan], 'priv_prop_covered': [np.nan], 'priv_prop_uncovered': [np.nan], 'mcaid_covered': [np.nan], 'mcaid_uncovered': [np.nan], 'mcaid_total': [np.nan], 'mcaid_prop_covered': [np.nan], 'mcaid_prop_uncovered': [np.nan], 'total_uninsured': [np.nan], 'prop_uninsured': [np.nan]}
ma_groups = pd.DataFrame(data=m)


index = 0
for year in range(2001, 2012):
    
    # Set year in treatment/control to current year
    control_groups.at[index, 'YEAR'] = year
    ma_groups.at[index, 'YEAR'] = year

    # Create new dataframes selecting for current year in treatment/control data
    control_year = control_states.loc[control_states['YEAR'] == year]
    ma_year = ma_data.loc[ma_data['YEAR'] == year]

    # How many people are covered by private insurance in each state (HCOVPRIV = 2)
    control_groups.at[index, 'priv_covered'] = control_year[control_year.HCOVPRIV == 2].shape[0]
    ma_groups.at[index, 'priv_covered'] = ma_year[ma_year.HCOVPRIV == 2].shape[0]

    # How many people are NOT covered by private insurance in each state (HCOVPRIV = 1)
    control_groups.at[index, 'priv_uncovered'] = control_year[control_year.HCOVPRIV == 1].shape[0]
    ma_groups.at[index, 'priv_uncovered'] = ma_year[ma_year.HCOVPRIV == 1].shape[0]

    # How many total people were sampled
    control_groups.at[index, 'priv_total'] = control_groups.at[index, 'priv_covered'] + control_groups.at[index, 'priv_uncovered']
    ma_groups.at[index, 'priv_total'] = ma_groups.at[index, 'priv_covered'] + ma_groups.at[index, 'priv_uncovered']

    # What is the proportion of people sampled who were covered by private insurance
    control_groups.at[index, 'priv_prop_covered'] = control_groups.at[index, 'priv_covered'] / control_groups.at[index, 'priv_total']
    ma_groups.at[index, 'priv_prop_covered'] = ma_groups.at[index, 'priv_covered'] / ma_groups.at[index, 'priv_total']

    # What is the proportion of people sampled who were NOT covered by private insurance
    control_groups.at[index, 'priv_prop_uncovered'] = control_groups.at[index, 'priv_uncovered'] / control_groups.at[index, 'priv_total']
    ma_groups.at[index, 'priv_prop_uncovered'] = ma_groups.at[index, 'priv_uncovered'] / ma_groups.at[index, 'priv_total']

    # How many people are covered by Medicare in each state (HIMCAIDLY = 2)
    control_groups.at[index, 'mcaid_covered'] = control_year[control_year.HIMCAIDLY == 2].shape[0]
    ma_groups.at[index, 'mcaid_covered'] = ma_year[ma_year.HIMCAIDLY == 2].shape[0]

    # How many people are NOT covered by Medicare in each state (HIMCAIDLY = 1)
    control_groups.at[index, 'mcaid_uncovered'] = control_year[control_year.HIMCAIDLY == 1].shape[0]
    ma_groups.at[index, 'mcaid_uncovered'] = ma_year[ma_year.HIMCAIDLY == 1].shape[0]

    # How many total people were sampled
    control_groups.at[index, 'mcaid_total'] = control_groups.at[index, 'mcaid_covered'] + control_groups.at[index, 'mcaid_uncovered']
    ma_groups.at[index, 'mcaid_total'] = ma_groups.at[index, 'mcaid_covered'] + ma_groups.at[index, 'mcaid_uncovered']

    # What is the proportion of people sampled who were covered by Medicaid
    control_groups.at[index, 'mcaid_prop_covered'] = control_groups.at[index, 'mcaid_covered'] / control_groups.at[index, 'mcaid_total']
    ma_groups.at[index, 'mcaid_prop_covered'] = ma_groups.at[index, 'mcaid_covered'] / ma_groups.at[index, 'mcaid_total']

    # What is the proportion of people sampled who were NOT covered by Medicaid
    control_groups.at[index, 'mcaid_prop_uncovered'] = control_groups.at[index, 'mcaid_uncovered'] / control_groups.at[index, 'mcaid_total']
    ma_groups.at[index, 'mcaid_prop_uncovered'] = ma_groups.at[index, 'mcaid_uncovered'] / ma_groups.at[index, 'mcaid_total']
    
    # How many people in the sample are verified to be uninsured (VERIFY = 2)
    control_groups.at[index, 'total_uninsured'] = control_year[control_year.HCOVANY == 1].shape[0]
    ma_groups.at[index, 'total_uninsured'] = ma_year[ma_year.HCOVANY == 1].shape[0]
    
    # What is the uninsurance rate (some samples are Not In Universe [NIU], hence why we cannot use the total number of individuals sampled)
    control_groups.at[index, 'prop_uninsured'] = control_groups.at[index, 'total_uninsured'] / (control_groups.at[index, 'total_uninsured'] + control_year[control_year.HCOVANY == 2].shape[0])
    ma_groups.at[index, 'prop_uninsured'] = ma_groups.at[index, 'total_uninsured'] / (ma_groups.at[index, 'total_uninsured'] + ma_year[ma_year.HCOVANY == 2].shape[0])

    index += 1


# In[13]:


control_groups


# In[14]:


ma_groups


# In[15]:


fig = plt.figure()

plt.plot(control_groups['YEAR'], control_groups['priv_prop_covered'], label='Control')
plt.plot(ma_groups['YEAR'], ma_groups['priv_prop_covered'], label='MA')
plt.legend(loc="upper right")
plt.title('Private Insurance Coverage Rates (2001-2011), MA vs. Control\n')
plt.xlabel('Year')
plt.ylabel('Private Insurance Coverage Rates')
plt.xlim(2001,2011)
plt.ylim(0,1)
plt.show()


# In[16]:


fig = plt.figure()

plt.plot(control_groups['YEAR'], control_groups['mcaid_prop_covered'], label='Control')
plt.plot(ma_groups['YEAR'], ma_groups['mcaid_prop_covered'], label='MA')
plt.legend(loc="upper right")
plt.title('Medicaid Coverage Rates (2001-2011), MA vs. Control\n')
plt.xlabel('Year')
plt.ylabel('Medicaid Coverage Rates')
plt.xlim(2001,2011)
plt.ylim(0,1)
plt.show()


# In[17]:


fig = plt.figure()

plt.plot(control_groups['YEAR'], control_groups['prop_uninsured'], label='Control')
plt.plot(ma_groups['YEAR'], ma_groups['prop_uninsured'], label='MA')
plt.legend(loc="upper right")
plt.title('Uninsurance Rates (2001-2011), MA vs. Control\n')
plt.xlabel('Year')
plt.ylabel('Uninsurance Rates')
plt.xlim(2001,2011)
plt.ylim(0,1)
plt.show()


# In[18]:


control_features = control_states[['YEAR', 'HCOVANY', 'HIMCAIDLY', 'HCOVPRIV', 'HEALTH', 'EMPSTAT', 'HHINCOME', 'AGE', 'SEX', 'RACE', 'MARST', 'HINSMIL', 'CITIZEN', 'HISPAN', 'POVERTY', 'EDUC', 'INCTOT', 'STATEFIP']]
control_features


# corrMatrix = control_features.corr()
# fig, ax = plt.subplots(figsize=(20,20))
# sns.heatmap(corrMatrix, annot=True, ax=ax)
# plt.show()

# In[19]:


ma_features = ma_data[['YEAR', 'HCOVANY', 'HIMCAIDLY', 'HCOVPRIV', 'HEALTH', 'EMPSTAT', 'HHINCOME', 'AGE', 'SEX', 'RACE', 'MARST', 'HINSMIL', 'CITIZEN', 'HISPAN', 'POVERTY', 'EDUC', 'INCTOT', 'STATEFIP']]
ma_features


# corrMatrix = ma_features.corr()
# fig, ax = plt.subplots(figsize=(20,20))
# sns.heatmap(corrMatrix, annot=True, ax=ax)
# plt.show()

# In[20]:


# Getting rid of NIU (0) and Armed Forces (1) values for EMPSTAT in control group
control_features = control_features[control_features.EMPSTAT != 0]
control_features = control_features[control_features.EMPSTAT != 1]
control_features["EMPSTAT"].value_counts()


# In[21]:


# We want to create two new dummy columns from EMPSTAT: whether the surveyed individual is in the labor force or not, and whether that individual currently holds a job 

def lab_force_dummy(x):
    if x == 10 or x == 21 or x == 12 or x == 22:
        return 1 # 1 denotes in labor force
    else:
        return 0 # 0 denotes not in labor force
    
def employment_dummy(x):
    if x == 10 or x == 12:
        return 1 # 1 denotes that this individual currently holds a job (employed)
    elif x == 21 or x == 22:
        return 0 # 0 denotes that this person does not currently hold a job (unemployed)
    else:
        return -1 # -1 denotes that this person is not in the labor force

control_lab_force = [lab_force_dummy(x) for x in control_features['EMPSTAT']]
control_employment = [employment_dummy(x) for x in control_features['EMPSTAT']]

control_features["LABFORCE"] = control_lab_force
control_features["EMPLOYED"] = control_employment
control_features


# In[22]:


# Repeat steps for Massachusetts

# Gets rid of NIU and Armed Forces values
ma_features = ma_features[ma_features.EMPSTAT != 0]
ma_features = ma_features[ma_features.EMPSTAT != 1]

ma_lab_force = [lab_force_dummy(x) for x in ma_features["EMPSTAT"]]
ma_employment = [employment_dummy(x) for x in ma_features["EMPSTAT"]]
ma_features["LABFORCE"] = ma_lab_force
ma_features["EMPLOYED"] = ma_employment
ma_features


# In[23]:


# We can now drop EMPSTAT from each table

control_features.drop("EMPSTAT", axis=1, inplace=True)
ma_features.drop("EMPSTAT", axis=1, inplace=True)


# In[24]:


control_features["RACE"].value_counts(normalize=True)


# In[25]:


ma_features["RACE"].value_counts(normalize=True)


# In[26]:


control_features["HISPAN"].value_counts(normalize=True)


# In[27]:


ma_features["HISPAN"].value_counts(normalize=True)


# In[28]:


# For the HISPAN variable, since the "Do not know" (901) and "No response" (902) make up less than 0.1% in both the control and MA feature datasets and only appear in 2001 and 2002, we will simply drop these features

control_features = control_features[control_features.HISPAN != 901]
control_features = control_features[control_features.HISPAN != 902]

ma_features = ma_features[ma_features.HISPAN != 901]
ma_features = ma_features[ma_features.HISPAN != 902]


# In[29]:


# Dummy encoding for race:
    # White (1) and Non-White (0)
    # Black (1) and Non-Black (0)
    # Native American (1) and Non-Native American (0)
    # Asian American and Pacific Islander (AAPI) (1) and non-AAPI (0)
    # Mixed-Race (1) and non-Mixed Race (0)
    # Hispanic (1) and non-Hispanic (0)

def white_dummy(x):
    if x == 100:
        return 1
    else:
        return 0

def aa_dummy(x):
    if x == 200:
        return 1
    else:
        return 0
    
def na_dummy(x):
    if x == 300:
        return 1
    else:
        return 0
    
def aapi_dummy(x):
    if x == 650 or x == 651 or x == 652:
        return 1
    else:
        return 0
    
def mixed_dummy(x):
    if x != 100 and x != 200 and x != 300 and x != 650 and x != 651 and x != 652:
        return 1
    else:
        return 0
    
def hispan_dummy(x):
    if x == 0:
        return 0
    else:
        return 1

    
    
# Control States
    
control_white = [white_dummy(x) for x in control_features["RACE"]]
control_aa = [aa_dummy(x) for x in control_features["RACE"]]
control_na = [na_dummy(x) for x in control_features["RACE"]]
control_aapi = [aapi_dummy(x) for x in control_features["RACE"]]
control_mixed = [mixed_dummy(x) for x in control_features["RACE"]]
control_hispan = [hispan_dummy(x) for x in control_features["HISPAN"]]

control_features["WHITE"] = control_white
control_features["BLACK"] = control_aa
control_features["NATIVE"] = control_na
control_features["AAPI"] = control_aapi
control_features["MIXED"] = control_mixed
control_features["HISP"] = control_hispan

control_features.drop("RACE", axis=1, inplace=True)
control_features.drop("HISPAN", axis=1, inplace=True)


# Massachusetts
    
ma_white = [white_dummy(x) for x in ma_features["RACE"]]
ma_aa = [aa_dummy(x) for x in ma_features["RACE"]]
ma_na = [na_dummy(x) for x in ma_features["RACE"]]
ma_aapi = [aapi_dummy(x) for x in ma_features["RACE"]]
ma_mixed = [mixed_dummy(x) for x in ma_features["RACE"]]
ma_hispan = [hispan_dummy(x) for x in ma_features["HISPAN"]]

ma_features["WHITE"] = ma_white
ma_features["BLACK"] = ma_aa
ma_features["NATIVE"] = ma_na
ma_features["AAPI"] = ma_aapi
ma_features["MIXED"] = ma_mixed
ma_features["HISP"] = ma_hispan

ma_features.drop("RACE", axis=1, inplace=True)
ma_features.drop("HISPAN", axis=1, inplace=True)


# In[30]:


control_features


# In[31]:


ma_features


# In[32]:


# Dummies for marital status - Married (1) and Unmarried (0)

def marital_dummy(x):
    if x == 1 or x == 2:
        return 1
    else:
        return 0
    
# Control
    
control_marital = [marital_dummy(x) for x in control_features["MARST"]]
control_features["MARITAL"] = control_marital
control_features.drop("MARST", axis=1, inplace=True)

# Massachusetts

ma_marital = [marital_dummy(x) for x in ma_features["MARST"]]
ma_features["MARITAL"] = ma_marital
ma_features.drop("MARST", axis=1, inplace=True)


# In[33]:


control_features


# In[34]:


ma_features


# In[35]:


# The MA Health Care Reform did not affect recipients of Medicare (Public healthcare for people aged 65+) and did not affect those in the Armed Forces

# We will limit each dataset to adults aged 21 to 64 to exclude recipients of Medicare or SCHIPLY

# We will also exclude individuals who receive Military or VA Public Healthcare since the MA Reform only expanded Medicaid coverage and not military insurance

control_features = control_features[control_features.HINSMIL == 1]
ma_features = ma_features[ma_features.HINSMIL == 1]

control_features = control_features[control_features.AGE >= 21]
control_features = control_features[control_features.AGE < 65]

ma_features = ma_features[ma_features.AGE >= 21]
ma_features = ma_features[ma_features.AGE < 65]


# In[36]:


control_features


# In[37]:


ma_features


# In[38]:


control_features.drop("HINSMIL", axis=1, inplace=True)
ma_features.drop("HINSMIL", axis=1, inplace=True)


# In[39]:


# Citizenship status: Citizen (1) vs. Non-Citizen (0)

def citizen_dummy(x):
    if x == 5:
        return 0
    else:
        return 1
    
# Control
    
control_citizen = [citizen_dummy(x) for x in control_features["CITIZEN"]]
control_features["CITZN"] = control_citizen
control_features.drop("CITIZEN", axis=1, inplace=True)

# Massachusetts

ma_citizen = [citizen_dummy(x) for x in ma_features["CITIZEN"]]
ma_features["CITZN"] = ma_citizen
ma_features.drop("CITIZEN", axis=1, inplace=True)


# In[40]:


control_features


# In[41]:


ma_features


# In[42]:


# Poverty status: Below the poverty line (1) and above the poverty line (0)

def poverty_dummy(x):
    if x == 10:
        return 1
    else:
        return 0
    
# Control
    
control_poverty = [poverty_dummy(x) for x in control_features["POVERTY"]]
control_features["POV"] = control_poverty
control_features.drop("POVERTY", axis=1, inplace=True)

# Massachusetts

ma_poverty = [poverty_dummy(x) for x in ma_features["POVERTY"]]
ma_features["POV"] = ma_poverty
ma_features.drop("POVERTY", axis=1, inplace=True)


# In[43]:


control_features


# In[44]:


ma_features


# In[45]:


control_features["EDUC"].value_counts(normalize=True)


# In[46]:


ma_features["EDUC"].value_counts(normalize=True)


# In[47]:


# Education dummies: high school diploma or above (1) and below high school diploma (0)

def educ_dummy(x):
    if x < 73:
        return 0
    else:
        return 1

# Control
    
control_educ = [educ_dummy(x) for x in control_features["EDUC"]]
control_features["EDU"] = control_educ
control_features.drop("EDUC", axis=1, inplace=True)

# Massachusetts

ma_educ = [educ_dummy(x) for x in ma_features["EDUC"]]
ma_features["EDU"] = ma_educ
ma_features.drop("EDUC", axis=1, inplace=True)


# In[48]:


control_features


# In[49]:


ma_features


# In[50]:


# Health Status Dummies: Good Health (1) and Bad Health (0)

def health_dummy(x):
    if x == 2 or x == 1:
        return 0
    else:
        return 1

# Control
    
control_health = [health_dummy(x) for x in control_features["HEALTH"]]
control_features["HSTAT"] = control_health
control_features.drop("HEALTH", axis=1, inplace=True)

# Massachusetts

ma_health = [health_dummy(x) for x in ma_features["HEALTH"]]
ma_features["HSTAT"] = ma_health
ma_features.drop("HEALTH", axis=1, inplace=True)


# In[51]:


control_features


# In[52]:


ma_features


# In[53]:


control_features_year = control_features.loc[control_features["YEAR"] == 2001].drop(["HIMCAIDLY", "HCOVPRIV"], axis=1).corr()
control_features_year


# In[54]:


control_correlation_time = pd.DataFrame([control_features_year.iloc[1, :].values.tolist()])
control_correlation_time


# In[55]:


for year in range(2002, 2012):
    control_features_year = control_features.loc[control_features["YEAR"] == year].drop(["HIMCAIDLY", "HCOVPRIV"], axis=1).corr()
    year_time = pd.DataFrame([control_features_year.iloc[1, :].values.tolist()])
    control_correlation_time = control_correlation_time.append(year_time, ignore_index=True)

control_correlation_time


# In[56]:


year = 2001
for i in range(11):
    control_correlation_time.iloc[i, 0] = year
    year += 1
    
control_correlation_time


# In[57]:


control_correlation_time.columns = list(control_features_year.columns)
control_correlation_time


# In[58]:


ma_features_year = ma_features.loc[ma_features["YEAR"] == 2001].drop(["HIMCAIDLY", "HCOVPRIV"], axis=1).corr()
ma_correlation_time = pd.DataFrame([ma_features_year.iloc[1, :].values.tolist()])
for year in range(2002, 2012):
    ma_features_year = ma_features.loc[ma_features["YEAR"] == year].drop(["HIMCAIDLY", "HCOVPRIV"], axis=1).corr()
    year_time = pd.DataFrame([ma_features_year.iloc[1, :].values.tolist()])
    ma_correlation_time = ma_correlation_time.append(year_time, ignore_index=True)

year = 2001
for i in range(11):
    ma_correlation_time.iloc[i, 0] = year
    year += 1

ma_correlation_time.columns = list(ma_features_year.columns)
ma_correlation_time


# In[59]:


def correlation_heatmap(df):
    correlations = df.corr()

    fig, ax = plt.subplots(figsize=(30,30))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
    plt.show();
    
correlation_heatmap(control_features)


# In[60]:


correlation_heatmap(ma_features)


# In[61]:


column_names = list(control_correlation_time.columns)
column_names


# In[62]:


for label in column_names:
    fig = plt.figure()

    plt.plot(control_correlation_time['YEAR'], control_correlation_time[label], label='Control')
    plt.plot(ma_correlation_time['YEAR'], ma_correlation_time[label], label='MA')
    plt.legend(loc="upper right")
    plt.title(f"Pearson correlation coefficient, Coverage vs. {label} 2001-2011\n")
    plt.xlabel('Year')
    plt.ylabel('Correlation Coefficient')
    plt.xlim(2001,2011)
    plt.ylim(0,1)
    plt.show()


# Selected covariates: HHINCOME, AGE, SEX, EMPLOYED, HISPAN, MARITAL, CITZN, POV, EDU

# In[63]:


import sys
get_ipython().system('{sys.executable} -m pip install SyntheticControlMethods')


# In[64]:


plt.hist(control_features['HHINCOME'], bins=range(0,750000,10000), color='green', edgecolor='black')


# In[65]:


plt.hist(control_features['AGE'], bins=range(20,65,5), color='green', edgecolor='black')


# In[66]:


plt.hist(ma_features['HHINCOME'], bins=range(0,750000,10000), color='purple', edgecolor='black')


# In[67]:


plt.hist(control_features['AGE'], bins=range(20,65,5), color='purple', edgecolor='black')


# In[68]:


synth_c = {'YEAR': [np.nan], 'STATEFIP': [np.nan], 'total_uninsured': [np.nan], 'prop_uninsured': [np.nan], 'priv_coverage': [np.nan], 'mcaid_coverage': [np.nan], 'health_stat': [np.nan], 'median_hhincome': [np.nan], 'mean_age': [np.nan], 'total_unemployed': [np.nan], 'u_rate': [np.nan], 'prop_fem': [np.nan], 'prop_hispan': [np.nan], 'prop_married': [np.nan], 'prop_ctzn': [np.nan], 'prop_pov': [np.nan], 'prop_edu': [np.nan]}
synth_control = pd.DataFrame(data=c)

synth_m = {'YEAR': [np.nan], 'STATEFIP': [np.nan], 'total_uninsured': [np.nan], 'prop_uninsured': [np.nan], 'priv_coverage': [np.nan], 'mcaid_coverage': [np.nan], 'health_stat': [np.nan], 'median_hhincome': [np.nan], 'mean_age': [np.nan], 'total_unemployed': [np.nan], 'u_rate': [np.nan], 'prop_fem': [np.nan], 'prop_hispan': [np.nan], 'prop_married': [np.nan], 'prop_ctzn': [np.nan], 'prop_pov': [np.nan], 'prop_edu': [np.nan]}
synth_ma = pd.DataFrame(data=m)

state_codes = control_features["STATEFIP"].unique().tolist()

index = 0

for state in state_codes:
    for year in range(2001, 2012):

        # Set year in control to current year
        synth_control.at[index, 'YEAR'] = year - 1
        
        # Set state FIP code to current state
        synth_control.at[index, 'STATEFIP'] = state

        # Create new dataframes selecting for current year in control data
        control_features_state = control_features.loc[control_features['STATEFIP'] == state]
        synth_control_year = control_features_state.loc[control_features_state['YEAR'] == year]

        # How many people in the sample are uninsured (HCOVANY = 1)
        synth_control.at[index, 'total_uninsured'] = synth_control_year[synth_control_year.HCOVANY == 1].shape[0]
        
        # What is the uninsurance rate
        synth_control.at[index, 'prop_uninsured'] = 100000 * synth_control.at[index, 'total_uninsured'] / (synth_control.at[index, 'total_uninsured'] + synth_control_year[synth_control_year.HCOVANY == 2].shape[0])
        
        # Share of people covered by private insurance (HCOVPRIV = 2)
        synth_control.at[index, 'priv_coverage'] = 100000 * synth_control_year[synth_control_year.HCOVPRIV == 2].shape[0] / synth_control_year.shape[0]
        
        # Share of people covered by Medicaid (HIMCAIDLY = 2)
        synth_control.at[index, 'mcaid_coverage'] = 100000 * synth_control_year[synth_control_year.HIMCAIDLY == 2].shape[0] / synth_control_year.shape[0]
        
        # Share of people who are in "good health" (HSTAT = 1)
        synth_control.at[index, 'health_stat'] = 100000 * synth_control_year[synth_control_year.HSTAT == 1].shape[0] / synth_control_year.shape[0]
        
        # What is the median household income
        synth_control.at[index, 'median_hhincome'] = synth_control_year['HHINCOME'].median()

        # What is the mean age
        synth_control.at[index, 'mean_age'] = synth_control_year['AGE'].mean()
        
        # What is the total number of people who are unemployed
        synth_control.at[index, 'total_unemployed'] = synth_control_year[synth_control_year.EMPLOYED == 0].shape[0]

        # What is the unemployment rate (Number of people unemloyed / size of labor force)
        synth_control.at[index, 'u_rate'] = synth_control.at[index, 'total_unemployed'] / (synth_control.at[index, 'total_unemployed'] + synth_control_year[synth_control_year.EMPLOYED == 1].shape[0])

        # Proportion of sample who are female
        synth_control.at[index, 'prop_fem'] = 100000 * synth_control_year[synth_control_year.SEX == 2].shape[0] / synth_control_year.shape[0]
        
        # Proportion of sample who are Hispanic
        synth_control.at[index, 'prop_hispan'] = 100000 * synth_control_year[synth_control_year.HISP == 1].shape[0] / synth_control_year.shape[0]

        # Proportion of sample who are married
        synth_control.at[index, 'prop_married'] = 100000 * synth_control_year[synth_control_year.MARITAL == 1].shape[0] / synth_control_year.shape[0]

        # Proportion of sample who are citizens
        synth_control.at[index, 'prop_ctzn'] = 100000 * synth_control_year[synth_control_year.CITZN == 1].shape[0] / synth_control_year.shape[0]

        # Proportion of sample who are in poverty (poverty rate)
        synth_control.at[index, 'prop_pov'] = 100000 * synth_control_year[synth_control_year.POV == 1].shape[0] / synth_control_year.shape[0]

        # Proportion of sample who have at least a high school diploma
        synth_control.at[index, 'prop_edu'] = 100000 * synth_control_year[synth_control_year.EDU == 1].shape[0] / synth_control_year.shape[0]

        index += 1

index = 0
for year in range(2001, 2012):
    
    # Set year in treatment to current year
    synth_ma.at[index, 'YEAR'] = year - 1
    
    # Set state FIP code to 25 (Massachusetts)
    synth_ma.at[index, 'STATEFIP'] = 25
    
    # Create new dataframes selecting for current state AND current year in treatment data
    synth_ma_year = ma_features.loc[ma_features['YEAR'] == year]
    
    # How many people in the sample are uninsured (HCOVANY = 1)
    synth_ma.at[index, 'total_uninsured'] = synth_ma_year[synth_ma_year.HCOVANY == 1].shape[0]
    
    # What is the uninsurance rate
    synth_ma.at[index, 'prop_uninsured'] = 100000 * synth_ma.at[index, 'total_uninsured'] / (synth_ma.at[index, 'total_uninsured'] + synth_ma_year[synth_ma_year.HCOVANY == 2].shape[0])
    
    # Share of people covered by private insurance (HCOVPRIV = 2)
    synth_ma.at[index, 'priv_coverage'] = 100000 * synth_ma_year[synth_ma_year.HCOVPRIV == 2].shape[0] / synth_ma_year.shape[0]
        
    # Share of people covered by Medicaid (HIMCAIDLY = 2)
    synth_ma.at[index, 'mcaid_coverage'] = 100000 * synth_ma_year[synth_ma_year.HIMCAIDLY == 2].shape[0] / synth_ma_year.shape[0]
    
    # Share of people who are in "Good Health" (HSTAT = 1)
    synth_ma.at[index, 'health_stat'] = 100000 * synth_ma_year[synth_ma_year.HSTAT == 1].shape[0] / synth_ma_year.shape[0]
    
    # What is the median household income
    synth_ma.at[index, 'median_hhincome'] = synth_ma_year['HHINCOME'].median()
    
    # What is the mean age
    synth_ma.at[index, 'mean_age'] = synth_ma_year['AGE'].mean()
    
    # What is the total number of people who are unemployed
    synth_ma.at[index, 'total_unemployed'] = synth_ma_year[synth_ma_year.EMPLOYED == 0].shape[0]
    
    # What is the unemployment rate (Number of people unemloyed / size of labor force)
    synth_ma.at[index, 'u_rate'] = synth_ma.at[index, 'total_unemployed'] / (synth_ma.at[index, 'total_unemployed'] + synth_ma_year[synth_ma_year.EMPLOYED == 1].shape[0])
    
    # Proportion of sample who are female
    synth_ma.at[index, 'prop_fem'] = 100000 * synth_ma_year[synth_ma_year.SEX == 2].shape[0] / synth_ma_year.shape[0]
    
    # Proportion of sample who are Hispanic
    synth_ma.at[index, 'prop_hispan'] = 100000 * synth_ma_year[synth_ma_year.HISP == 1].shape[0] / synth_ma_year.shape[0]
    
    # Proportion of sample who are married
    synth_ma.at[index, 'prop_married'] = 100000 * synth_ma_year[synth_ma_year.MARITAL == 1].shape[0] / synth_ma_year.shape[0]
    
    # Proportion of sample who are citizens
    synth_ma.at[index, 'prop_ctzn'] = 100000 * synth_ma_year[synth_ma_year.CITZN == 1].shape[0] / synth_ma_year.shape[0]
    
    # Proportion of sample who are in poverty (poverty rate)
    synth_ma.at[index, 'prop_pov'] = 100000 * synth_ma_year[synth_ma_year.POV == 1].shape[0] / synth_ma_year.shape[0]
    
    # Proportion of sample who have at least a high school diploma
    synth_ma.at[index, 'prop_edu'] = 100000 * synth_ma_year[synth_ma_year.EDU == 1].shape[0] / synth_ma_year.shape[0]
    
    index += 1


# In[69]:


synth_control.dropna(axis=1, inplace=True)
synth_control.head(n=20)


# In[70]:


synth_ma.dropna(axis=1, inplace=True)
synth_ma


# In[71]:


full_synth_data = synth_control.append(synth_ma)
full_synth_data.drop(['total_uninsured', 'total_unemployed'], axis=1, inplace=True)
full_synth_data


# In[72]:


full_synth_data.reset_index(drop=True, inplace=True)


# In[73]:


# full_synth_data.to_csv('full_synth_data.csv', index=False, encoding='utf-8')


# In[74]:


from statsmodels.formula.api import ols

model_all = ols('prop_uninsured ~ median_hhincome + mean_age + u_rate + prop_hispan + prop_married + prop_ctzn + prop_pov + prop_edu', data=full_synth_data).fit()
print(model_all.summary())

model_priv = ols('priv_coverage ~ median_hhincome + mean_age + u_rate + prop_hispan + prop_married + prop_ctzn + prop_pov + prop_edu', data=full_synth_data).fit()
print(model_priv.summary())

model_mcaid = ols('mcaid_coverage ~ median_hhincome + mean_age + u_rate + prop_hispan + prop_married + prop_ctzn + prop_pov + prop_edu', data=full_synth_data).fit()
print(model_mcaid.summary())


# In[75]:


model_all = ols('prop_uninsured ~ median_hhincome + mean_age + u_rate + prop_hispan + prop_married + prop_pov + prop_edu', data=full_synth_data).fit()
print(model_all.summary())

model_priv = ols('priv_coverage ~ median_hhincome + mean_age + u_rate + prop_hispan + prop_pov + prop_edu', data=full_synth_data).fit()
print(model_priv.summary())

model_mcaid = ols('mcaid_coverage ~ median_hhincome + mean_age + prop_married + prop_pov + prop_edu', data=full_synth_data).fit()
print(model_mcaid.summary())


# In[76]:


health_data_full = pd.read_csv("crude_mortality_rates.csv")
health_data_full


# In[77]:


health_data_full.dropna(axis=0, inplace=True)
health_data_full


# In[78]:


health_data_full = health_data_full.loc[health_data_full['State'] != 'Oregon']
health_data_full = health_data_full.loc[health_data_full['State'] != 'Hawaii']
health_data_full = health_data_full.rename(columns={"Year": "YEAR", "State Code": "STATEFIP", "Crude Rate": "crude_rate"})
health_data_full


# In[79]:


health_synth_data = full_synth_data.merge(health_data_full)
health_synth_data = health_synth_data.drop(['State', 'Year Code', 'Deaths', 'Population'], axis=1)
health_synth_data


# In[80]:


from statsmodels.formula.api import ols

model_all = ols('crude_rate ~ median_hhincome + mean_age + u_rate + prop_hispan + prop_married + prop_ctzn + prop_pov + prop_edu', data=health_synth_data).fit()
print(model_all.summary())


# In[81]:


from statsmodels.formula.api import ols

model_all = ols('crude_rate ~ median_hhincome + mean_age + prop_hispan + prop_married + prop_ctzn + prop_pov + prop_edu', data=health_synth_data).fit()
print(model_all.summary())


# In[82]:


health_synth_data.to_csv('full_synth_data.csv', index=False, encoding='utf-8')


# In[ ]:




