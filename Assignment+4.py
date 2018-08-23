
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._
# 
# ---

# In[11]:

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind


# # Assignment 4 - Hypothesis Testing
# This assignment requires more individual learning than previous assignments - you are encouraged to check out the [pandas documentation](http://pandas.pydata.org/pandas-docs/stable/) to find functions or methods you might not have used yet, or ask questions on [Stack Overflow](http://stackoverflow.com/) and tag them as pandas and python related. And of course, the discussion forums are open for interaction with your peers and the course staff.
# 
# Definitions:
# * A _quarter_ is a specific three month period, Q1 is January through March, Q2 is April through June, Q3 is July through September, Q4 is October through December.
# * A _recession_ is defined as starting with two consecutive quarters of GDP decline, and ending with two consecutive quarters of GDP growth.
# * A _recession bottom_ is the quarter within a recession which had the lowest GDP.
# * A _university town_ is a city which has a high percentage of university students compared to the total population of the city.
# 
# **Hypothesis**: University towns have their mean housing prices less effected by recessions. Run a t-test to compare the ratio of the mean price of houses in university towns the quarter before the recession starts compared to the recession bottom. (`price_ratio=quarter_before_recession/recession_bottom`)
# 
# The following data files are available for this assignment:
# * From the [Zillow research data site](http://www.zillow.com/research/data/) there is housing data for the United States. In particular the datafile for [all homes at a city level](http://files.zillowstatic.com/research/public/City/City_Zhvi_AllHomes.csv), ```City_Zhvi_AllHomes.csv```, has median home sale prices at a fine grained level.
# * From the Wikipedia page on college towns is a list of [university towns in the United States](https://en.wikipedia.org/wiki/List_of_college_towns#College_towns_in_the_United_States) which has been copy and pasted into the file ```university_towns.txt```.
# * From Bureau of Economic Analysis, US Department of Commerce, the [GDP over time](http://www.bea.gov/national/index.htm#gdp) of the United States in current dollars (use the chained value in 2009 dollars), in quarterly intervals, in the file ```gdplev.xls```. For this assignment, only look at GDP data from the first quarter of 2000 onward.
# 
# Each function in this assignment below is worth 10%, with the exception of ```run_ttest()```, which is worth 50%.

# In[12]:

# Use this dictionary to map state names to two letter acronyms
import pandas as pd
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}


# In[13]:

def get_list_of_university_towns():
    '''Returns a DataFrame of towns and the states they are in from the 
    university_towns.txt list. The format of the DataFrame should be:
    DataFrame( [ ["Michigan", "Ann Arbor"], ["Michigan", "Yipsilanti"] ], 
    columns=["State", "RegionName"]  )
    
    The following cleaning needs to be done:

    1. For "State", removing characters from "[" to the end.
    2. For "RegionName", when applicable, removing every character from " (" to the end.
    3. Depending on how you read the data, you may need to remove newline character '\n'. '''
    import pandas as pd
    university_towns = pd.read_table('university_towns.txt', delimiter=None, header=None)
    states=university_towns[university_towns[0].str.match('.*\[edit\]$')]                                            .replace(to_replace='\n$',value='', regex=True)                                            .replace(to_replace='\[.*',value='', regex=True)
    towns=university_towns[~university_towns[0].str.match('.*\[edit\]$')]                             .replace(to_replace='\[.*\]',value='', regex=True)                             .replace('\(.*','', regex=True)                             .replace('\n$','', regex=True)                             .replace('\s+$','', regex=True)
    df=pd.DataFrame()
    df['RegionName']=towns[0]
    df['States']=None
    
    '''i=0
    for j in range(len(states.index)):
        if j==len(states.index)-1:
            k=len(university_towns[0])
        else:
            k=states.index[j+1]
        num_of_records=k-states.index[j]
        df.iloc[i:i+num_of_records-1,1]=states.iloc[j,0]
        i = i + num_of_records - 1
        df.columns=['RegionName','State']'''
    for i in states.index:
        df.loc[i+1,'State']=states.loc[i,0]

    df=df.fillna(method='ffill')
    df.reset_index(inplace=True)
    return df[['State','RegionName']]





# In[14]:

def get_recession_periods():
    gdp=pd.read_excel('gdplev.xls',skiprows=219,header=None).drop([0,1,2,3,5,7],axis=1).rename(columns={4:'quarter',6:'gdp'})
    g=gdp['gdp']
    possible_start=None
    possible_bottom=None
    recession_periods=[]
    for i in range(1,len(g)):
        if possible_start == None and i < len(g)-1 and g[i] < g[i-1] and g[i] > g[i+1]:
            possible_start = i;
            possible_bottom = i+1;
            i=i+1
            continue
            
        if possible_bottom != None and g[i] < g[possible_bottom]:
            possible_bottom = i
            
        if i < len(g)-1 and possible_start != None and g[i] > g[i-1] and g[i] < g[i+1]:
            recession_periods.append((gdp.iloc[possible_start,0], gdp.iloc[i+1,0],gdp.iloc[possible_bottom,0]))
            i=i+1
            possible_start=possible_bottom=None
    return recession_periods        
    

def get_recession_start():
    '''Returns the year and quarter of the recession start time as a 
    string value in a format such as 2005q3'''
    
    return get_recession_periods()[0][0]

#get_recession_periods()


# In[15]:

def get_recession_end():
    '''Returns the year and quarter of the recession end time as a 
    string value in a format such as 2005q3'''
       
    return get_recession_periods()[0][1]
#get_recession_end()


# In[16]:

def get_recession_bottom():
    '''Returns the year and quarter of the recession bottom time as a 
    string value in a format such as 2005q3'''
    
    return get_recession_periods()[0][2]
#get_recession_bottom()


# In[17]:

ncols=[str(i)+j for i in range(2000,2016) for j in ['q1','q2','q3','q4']] + ['2016q1','2016q2']
def convert_housing_data_to_quarters():
    '''Converts the housing data to quarters and returns it as mean 
    values in a dataframe. This dataframe should be a dataframe with
    columns for 2000q1 through 2016q3, and should have a multi-index
    in the shape of ["State","RegionName"].
    
    Note: Quarters are defined in the assignment description, they are
    not arbitrary three month periods.
    
    The resulting dataframe should have 67 columns, and 10,730 rows.'''
    
    homes=pd.read_csv('City_Zhvi_AllHomes.csv')
    columns=[col for col in homes.columns if col.startswith('20')][:-2]
    df=homes[columns].apply(foo1,axis=1)
    df['2016q3']=np.mean(homes['2016-07'] + homes['2016-08'])
    df['State']=homes['State'].apply(lambda v : states.get(v))
    df['RegionName']=homes['RegionName']
    df=df.set_index(['State','RegionName'])
    
    '''columns=['State','RegionName'] + [col for col in homes.columns if col.startswith('20')]
    homes['State']=homes['State'].apply(lambda v : states.get(v))
    #print(homes.head()[columns])
    df=homes[columns].set_index(['State','RegionName']).apply(foo,axis=1)'''
    return df

def foo(row):
    res={}
    #print('calling')
    i=0
    for i in range(0,len(row.index),3):
        l = len(row.index)
        name = row.index[i]
        #print(name)
        month = int(name.split('-')[1])
        year = name.split('-')[0]
        if month == 1:
            res[year+'q1']=np.mean(row[i:i+3])
        elif month == 4:
            res[year+'q2']=np.mean(row[i:i+3])
        elif month == 7:
            res[year+'q3']=np.mean(row[i:i+3])
        elif month == 10:
            res[year+'q4']=np.mean(row[i:i+3])

    
    return pd.Series(res)
        
def foo1(row):
    arr=row.reshape(len(row)/3,3)
    arr=arr.mean(axis=1)
    return pd.Series(arr, index=ncols)

convert_housing_data_to_quarters()
#homes=pd.read_csv('City_Zhvi_AllHomes.csv')
#print(convert_housing_data_to_quarters().head(2))
#homes.head(2)[['2000-01','2000-02','2000-03']]


# In[1]:

def run_ttest():
    '''First creates new data showing the decline or growth of housing prices
    between the recession start and the recession bottom. Then runs a ttest
    comparing the university town values to the non-university towns values, 
    return whether the alternative hypothesis (that the two groups are the same)
    is true or not as well as the p-value of the confidence. 
    
    Return the tuple (different, p, better) where different=True if the t-test is
    True at a p<0.01 (we reject the null hypothesis), or different=False if 
    otherwise (we cannot reject the null hypothesis). The variable p should
    be equal to the exact p value returned from scipy.stats.ttest_ind(). The
    value for better should be either "university town" or "non-university town"
    depending on which has a lower mean price ratio (which is equivilent to a
    reduced market loss).'''
    rec_start = before(get_recession_start())
    rec_bottom = get_recession_bottom()
    housing_data = convert_housing_data_to_quarters()[[rec_start,rec_bottom]]
    housing_data['price_ratio']=housing_data.apply(lambda row: row[rec_start]/row[rec_bottom],axis=1)
    u_towns=get_list_of_university_towns()
    u_towns=u_towns.set_index(['State','RegionName'])
    housing_data_of_u_towns = housing_data.loc[u_towns.index.intersection(housing_data.index)]['price_ratio']
    housing_data_of_nu_towns = housing_data.loc[housing_data.index.difference(u_towns.index)]['price_ratio']
    #housing_data_of_u_towns=housing_data_of_u_towns.groupby(level=[0,1]).mean()
    #housing_data_of_nu_towns=housing_data_of_nu_towns.groupby(level=[0,1]).mean()
    from scipy import stats
    s,p=stats.ttest_ind(housing_data_of_nu_towns, housing_data_of_u_towns,nan_policy='omit')
    return (p<0.01,p,'university town' if housing_data_of_u_towns.mean() < housing_data_of_nu_towns.mean() else 'non-university town')

def before(r):
    year,quarter = int(r.split('q')[0]),int(r.split('q')[1])
    if quarter == 1:
        return '{}q{}'.format(year-1,4)
    else:
        return '{}q{}'.format(year,quarter-1)
    
run_ttest()


# In[ ]:



