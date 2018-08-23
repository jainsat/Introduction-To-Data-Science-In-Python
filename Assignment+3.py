
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.5** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._
# 
# ---

# # Assignment 3 - More Pandas
# This assignment requires more individual learning then the last one did - you are encouraged to check out the [pandas documentation](http://pandas.pydata.org/pandas-docs/stable/) to find functions or methods you might not have used yet, or ask questions on [Stack Overflow](http://stackoverflow.com/) and tag them as pandas and python related. And of course, the discussion forums are open for interaction with your peers and the course staff.

# ### Question 1 (20%)
# Load the energy data from the file `Energy Indicators.xls`, which is a list of indicators of [energy supply and renewable electricity production](Energy%20Indicators.xls) from the [United Nations](http://unstats.un.org/unsd/environment/excel_file_tables/2013/Energy%20Indicators.xls) for the year 2013, and should be put into a DataFrame with the variable name of **energy**.
# 
# Keep in mind that this is an Excel file, and not a comma separated values file. Also, make sure to exclude the footer and header information from the datafile. The first two columns are unneccessary, so you should get rid of them, and you should change the column labels so that the columns are:
# 
# `['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']`
# 
# Convert `Energy Supply` to gigajoules (there are 1,000,000 gigajoules in a petajoule). For all countries which have missing data (e.g. data with "...") make sure this is reflected as `np.NaN` values.
# 
# Rename the following list of countries (for use in later questions):
# 
# ```"Republic of Korea": "South Korea",
# "United States of America": "United States",
# "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
# "China, Hong Kong Special Administrative Region": "Hong Kong"```
# 
# There are also several countries with numbers and/or parenthesis in their name. Be sure to remove these, 
# 
# e.g. 
# 
# `'Bolivia (Plurinational State of)'` should be `'Bolivia'`, 
# 
# `'Switzerland17'` should be `'Switzerland'`.
# 
# <br>
# 
# Next, load the GDP data from the file `world_bank.csv`, which is a csv containing countries' GDP from 1960 to 2015 from [World Bank](http://data.worldbank.org/indicator/NY.GDP.MKTP.CD). Call this DataFrame **GDP**. 
# 
# Make sure to skip the header, and rename the following list of countries:
# 
# ```"Korea, Rep.": "South Korea", 
# "Iran, Islamic Rep.": "Iran",
# "Hong Kong SAR, China": "Hong Kong"```
# 
# <br>
# 
# Finally, load the [Sciamgo Journal and Country Rank data for Energy Engineering and Power Technology](http://www.scimagojr.com/countryrank.php?category=2102) from the file `scimagojr-3.xlsx`, which ranks countries based on their journal contributions in the aforementioned area. Call this DataFrame **ScimEn**.
# 
# Join the three datasets: GDP, Energy, and ScimEn into a new dataset (using the intersection of country names). Use only the last 10 years (2006-2015) of GDP data and only the top 15 countries by Scimagojr 'Rank' (Rank 1 through 15). 
# 
# The index of this DataFrame should be the name of the country, and the columns should be ['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations',
#        'Citations per document', 'H index', 'Energy Supply',
#        'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008',
#        '2009', '2010', '2011', '2012', '2013', '2014', '2015'].
# 
# *This function should return a DataFrame with 20 columns and 15 entries.*

# In[2]:


def answer_one():
    import pandas as pd
    import numpy as np
    energy = pd.read_excel('Energy Indicators.xls', skiprows=17, skip_footer=38)             .drop(['Unnamed: 0', 'Unnamed: 1'], axis=1)             .rename(columns={'Unnamed: 2': 'Country', 'Petajoules': 'Energy Supply', 'Gigajoules':'Energy Supply per Capita', '%':'% Renewable'}) 

 
    energy['Energy Supply'] *= 1000000
    energy['Country']=energy['Country'].str.replace('[0-9]+', '')
    energy['Country']=energy['Country'].str.replace('\(.*\)', '')
    energy['Country']=energy['Country'].str.replace('\s+$', '')

    energy=energy.replace({'...':np.NaN,                           'Republic of Korea': 'South Korea',                           'United States of America': 'United States',                           'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',                           'China, Hong Kong Special Administrative Region': 'Hong Kong'})  
                    


    columns=[str(i) for i in np.arange(2006, 2016)]
    columns.insert(0,'Country Name')
    gdp=pd.read_csv('world_bank.csv',skiprows=4).replace({"Korea, Rep.": "South Korea", 
                                                          "Iran, Islamic Rep.": "Iran",
                                                          "Hong Kong SAR, China": "Hong Kong"})[columns].rename(columns={'Country Name':'Country'})
    scimen=pd.read_excel('scimagojr-3.xlsx').head(15)

    df=pd.merge(pd.merge(energy, scimen, how='inner', left_on='Country', right_on='Country'), gdp,how='inner', left_on='Country', right_on='Country')
    df=df[['Country','Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations', 'Citations per document', 'H index', 'Energy Supply', 'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']]
    df=df.set_index('Country')
    df['Energy Supply']=df['Energy Supply'].astype('float64')
    return df


# ### Question 2 (6.6%)
# The previous question joined three datasets then reduced this to just the top 15 entries. When you joined the datasets, but before you reduced this to the top 15 items, how many entries did you lose?
# 
# *This function should return a single number.*

# In[3]:

get_ipython().run_cell_magic('HTML', '', '<svg width="800" height="300">\n  <circle cx="150" cy="180" r="80" fill-opacity="0.2" stroke="black" stroke-width="2" fill="blue" />\n  <circle cx="200" cy="100" r="80" fill-opacity="0.2" stroke="black" stroke-width="2" fill="red" />\n  <circle cx="100" cy="100" r="80" fill-opacity="0.2" stroke="black" stroke-width="2" fill="green" />\n  <line x1="150" y1="125" x2="300" y2="150" stroke="black" stroke-width="2" fill="black" stroke-dasharray="5,3"/>\n  <text  x="300" y="165" font-family="Verdana" font-size="35">Everything but this!</text>\n</svg>')


# In[4]:

def answer_two():
    import pandas as pd
    import numpy as np
    energy = pd.read_excel('Energy Indicators.xls', skiprows=17, skip_footer=38)             .drop(['Unnamed: 0', 'Unnamed: 1'], axis=1)             .rename(columns={'Unnamed: 2': 'Country', 'Petajoules': 'Energy Supply', 'Gigajoules':'Energy Supply per Capita', '%':'% Renewable'}) 

 
    energy['Energy Supply'] *= 1000000
    energy['Country']=energy['Country'].str.replace('[0-9]+', '')
    energy['Country']=energy['Country'].str.replace('\(.*\)', '')
    energy['Country']=energy['Country'].str.replace('\s+$', '')

    energy=energy.replace({'...':np.NaN,                           'Republic of Korea': 'South Korea',                           'United States of America': 'United States',                           'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',                           'China, Hong Kong Special Administrative Region': 'Hong Kong'})  
                    


    columns=[str(i) for i in np.arange(2006, 2016)]
    columns.insert(0,'Country Name')
    gdp=pd.read_csv('world_bank.csv',skiprows=4).replace({"Korea, Rep.": "South Korea", 
                                                          "Iran, Islamic Rep.": "Iran",
                                                          "Hong Kong SAR, China": "Hong Kong"})[columns].rename(columns={'Country Name':'Country'})
    scimen=pd.read_excel('scimagojr-3.xlsx')
    intersection=pd.merge(pd.merge(energy, scimen, how='inner', left_on='Country', right_on='Country'), gdp,how='inner', left_on='Country', right_on='Country')
    union=pd.merge(pd.merge(energy, scimen, how='outer', left_on='Country', right_on='Country'), gdp,how='outer', left_on='Country', right_on='Country')
    return len(union.index) - len(intersection.index)

answer_two()


# ## Answer the following questions in the context of only the top 15 countries by Scimagojr Rank (aka the DataFrame returned by `answer_one()`)

# ### Question 3 (6.6%)
# What is the average GDP over the last 10 years for each country? (exclude missing values from this calculation.)
# 
# *This function should return a Series named `avgGDP` with 15 countries and their average GDP sorted in descending order.*

# In[5]:

def answer_three():
    import numpy as np
    df = answer_one()
    df = answer_one()
    columns=[str(i) for i in range(2006,2016)]
    res=df.apply(lambda row: np.mean(row[columns]), axis=1)
    res.name='avgGDP'    
    return res.sort_values(ascending=False)


# ### Question 4 (6.6%)
# By how much had the GDP changed over the 10 year span for the country with the 6th largest average GDP?
# 
# *This function should return a single number.*

# In[6]:

def answer_four():
    df=answer_one()
    country=answer_three().index[5]    
    return df.loc[country, '2015']-df.loc[country, '2006']


# ### Question 5 (6.6%)
# What is the mean `Energy Supply per Capita`?
# 
# *This function should return a single number.*

# In[7]:

def answer_five():
    import numpy as np
    df = answer_one()
    return np.mean(df['Energy Supply per Capita'])


# ### Question 6 (6.6%)
# What country has the maximum % Renewable and what is the percentage?
# 
# *This function should return a tuple with the name of the country and the percentage.*

# In[8]:

def answer_six():
    import numpy as np
    df = answer_one()
    return (np.argmax(df['% Renewable']), np.max(df['% Renewable']))


# ### Question 7 (6.6%)
# Create a new column that is the ratio of Self-Citations to Total Citations. 
# What is the maximum value for this new column, and what country has the highest ratio?
# 
# *This function should return a tuple with the name of the country and the ratio.*

# In[9]:

def answer_seven():
    import numpy as np
    df=answer_one()
    df['ratio citation']=df['Self-citations']/df['Citations']
    return (np.argmax(df['ratio citation']), np.max(df['ratio citation']))


# ### Question 8 (6.6%)
# 
# Create a column that estimates the population using Energy Supply and Energy Supply per capita. 
# What is the third most populous country according to this estimate?
# 
# *This function should return a single string value.*

# In[10]:

def answer_eight():
    df=answer_one()
    df['Population']=df['Energy Supply'] / df['Energy Supply per Capita']
    res=df['Population'].sort_values(ascending=False)
    return res.index[2] 


# ### Question 9 (6.6%)
# Create a column that estimates the number of citable documents per person. 
# What is the correlation between the number of citable documents per capita and the energy supply per capita? Use the `.corr()` method, (Pearson's correlation).
# 
# *This function should return a single number.*
# 
# *(Optional: Use the built-in function `plot9()` to visualize the relationship between Energy Supply per Capita vs. Citable docs per Capita)*

# In[11]:

def answer_nine():
    df=answer_one()
    df['Population']=df['Energy Supply'] / df['Energy Supply per Capita']
    df['Citable documents per Capita']=df['Citable documents'] / df['Population']
    return df[['Citable documents per Capita', 'Energy Supply per Capita']].corr().iloc[0,1]



# In[12]:

#def plot9():
 #   import matplotlib as plt
  #  %matplotlib inline
    
   # Top15 = answer_one()
    #Top15['PopEst'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    #Top15['Citable docs per Capita'] = Top15['Citable documents'] / Top15['PopEst']
    #Top15.plot(x='Citable docs per Capita', y='Energy Supply per Capita', kind='scatter', xlim=[0, 0.0006])


# In[13]:

#plot9() # Be sure to comment out plot9() before submitting the assignment!


# ### Question 10 (6.6%)
# Create a new column with a 1 if the country's % Renewable value is at or above the median for all countries in the top 15, and a 0 if the country's % Renewable value is below the median.
# 
# *This function should return a series named `HighRenew` whose index is the country name sorted in ascending order of rank.*

# In[14]:

def foo(row, *args):
    return 1 if row['% Renewable'] >= args[0] else 0
    
def answer_ten():
    import numpy as np
    df=answer_one().sort_values(by='Rank')
    med = np.median(df['% Renewable'])
    res=df.apply(foo, axis=1, args=(med,))
    res.name='HighRenew'
    return res


# ### Question 11 (6.6%)
# Use the following dictionary to group the Countries by Continent, then create a dateframe that displays the sample size (the number of countries in each continent bin), and the sum, mean, and std deviation for the estimated population of each country.
# 
# ```python
# ContinentDict  = {'China':'Asia', 
#                   'United States':'North America', 
#                   'Japan':'Asia', 
#                   'United Kingdom':'Europe', 
#                   'Russian Federation':'Europe', 
#                   'Canada':'North America', 
#                   'Germany':'Europe', 
#                   'India':'Asia',
#                   'France':'Europe', 
#                   'South Korea':'Asia', 
#                   'Italy':'Europe', 
#                   'Spain':'Europe', 
#                   'Iran':'Asia',
#                   'Australia':'Australia', 
#                   'Brazil':'South America'}
# ```
# 
# *This function should return a DataFrame with index named Continent `['Asia', 'Australia', 'Europe', 'North America', 'South America']` and columns `['size', 'sum', 'mean', 'std']`*

# In[15]:

ContinentDict  = {'China':'Asia', 
                  'United States':'North America', 
                  'Japan':'Asia', 
                  'United Kingdom':'Europe', 
                  'Russian Federation':'Europe', 
                  'Canada':'North America', 
                  'Germany':'Europe', 
                  'India':'Asia',
                  'France':'Europe', 
                  'South Korea':'Asia', 
                  'Italy':'Europe', 
                  'Spain':'Europe', 
                  'Iran':'Asia',
                  'Australia':'Australia', 
                  'Brazil':'South America'}
def answer_eleven():
    import pandas as pd
    import numpy as np
    df=answer_one()
    cdf=pd.DataFrame(ContinentDict, index=['Continent']).T
    cdf.index.name='Country'
    df=pd.merge(df,cdf,left_index=True,right_index=True,how='inner')
    df['Country']=df.index
    df['Population']=df['Energy Supply'] / df['Energy Supply per Capita']
    #print(df)
    res=df.groupby('Continent').agg({'Country':np.count_nonzero, 'Population':[np.sum, np.mean, np.std]})
    res.columns=res.columns.droplevel()
    res=res.rename(columns={'count_nonzero':'size'})
    return res

answer_eleven()


# ### Question 12 (6.6%)
# Cut % Renewable into 5 bins. Group Top15 by the Continent, as well as these new % Renewable bins. How many countries are in each of these groups?
# 
# *This function should return a __Series__ with a MultiIndex of `Continent`, then the bins for `% Renewable`. Do not include groups with no countries.*

# In[28]:

def answer_twelve():
    import pandas as pd
    import numpy as np
    df=answer_one()
    cdf=pd.DataFrame(ContinentDict, index=['Continent']).T
    cdf.index.name='Country'
    df=pd.merge(df,cdf,left_index=True,right_index=True,how='inner')[[ 'Continent', '% Renewable']]
    df['% Renewable']=pd.cut(df['% Renewable'],5)
    df=df.reset_index().set_index([ 'Continent','% Renewable']).groupby(level=[0,1]).agg({'Country':np.count_nonzero})
    return df['Country']

answer_twelve()


# ### Question 13 (6.6%)
# Convert the Population Estimate series to a string with thousands separator (using commas). Do not round the results.
# 
# e.g. 317615384.61538464 -> 317,615,384.61538464
# 
# *This function should return a Series `PopEst` whose index is the country name and whose values are the population estimate string.*

# In[27]:

def answer_thirteen():
    df=answer_one()
    df['Population']= list(map(lambda x : '{:,}'.format(x), df['Energy Supply'] / df['Energy Supply per Capita']))
    res=df['Population']
    res.name='PopEst'
    return res


# ### Optional
# 
# Use the built in function `plot_optional()` to see an example visualization.

# In[18]:

def plot_optional():
    import matplotlib as plt
    get_ipython().magic('matplotlib inline')
    Top15 = answer_one()
    ax = Top15.plot(x='Rank', y='% Renewable', kind='scatter', 
                    c=['#e41a1c','#377eb8','#e41a1c','#4daf4a','#4daf4a','#377eb8','#4daf4a','#e41a1c',
                       '#4daf4a','#e41a1c','#4daf4a','#4daf4a','#e41a1c','#dede00','#ff7f00'], 
                    xticks=range(1,16), s=6*Top15['2014']/10**10, alpha=.75, figsize=[16,6]);

    for i, txt in enumerate(Top15.index):
        ax.annotate(txt, [Top15['Rank'][i], Top15['% Renewable'][i]], ha='center')

    print("This is an example of a visualization that can be created to help understand the data. This is a bubble chart showing % Renewable vs. Rank. The size of the bubble corresponds to the countries' 2014 GDP, and the color corresponds to the continent.")


# In[19]:

#plot_optional() # Be sure to comment out plot_optional() before submitting the assignment!

