#!/usr/bin/env python
# coding: utf-8

# # 0) Environment Setup and Import

# In[22]:


import pandas as pd
import numpy as np
import urllib.parse as urlparse

try:
    import xlrd
except ImportError:
    get_ipython().system('pip install xlrd')


# In[9]:


df = pd.read_csv('./test.csv')
df = df[:200]


# # 1) Parser Function Setup

# In[10]:


def primaryDomain(x,l='len'):
    temp = urlparse.urlparse(x).hostname
    if l == 'len':
        return len(temp)
    else:
        return temp


# In[11]:


#Valid extension checker of domain names to allow for correct parsing of domain and extension
validExtensions = pd.read_excel("./Extensions.xlsx")
validExtensions = list(validExtensions['Ext'])


# In[13]:


#Extension Finder function that seperates extension from Primary Domain
def extFind(url,combine=True):
    t = primaryDomain(url,l='str')
    orig = t
    t = t.split(".")
    temp = ""
    last = t[-1]
    while (len(last) <= 3) & (last in validExtensions):
        if temp == "":
            temp = t.pop()
        else:
            temp = t.pop() + "." + temp
        try:
            last = t[-1]
        except IndexError:
            break
    if combine == True:
        if orig[:2] =="ww":
            t = [t[0],".".join(t[1:])]
        else:
            t = [".".join(t)]
    t.append(temp)
    #print(str(orig) + " --> " + str(t))
    return t


# In[17]:


#Parsing of domain into list and stored temporairly in Domain Temp to be used by urlVars function
df['domain_temp'] = ""
df["domain_temp"] = df.url.apply(lambda row: extFind(row,True))


# In[18]:


#Main feature creation function
def urlVars(df,domain_list_col):
#     df['Extension'] = df[domain_list_col].apply(lambda row: row[-1])
#     df['Extension_Length'] = df['Extension'].apply(lambda row: len(row))
#     df['Num_Hyphen'] = df.url.apply(lambda row: row.count("-"))
    
#     df['Longest_Word'] = df.Word_Counts.apply(lambda row: max(row))
#     df['Shortest_Word'] = df.Word_Counts.apply(lambda row: min(row))
#     df['Mean_Word'] = df.Word_Counts.apply(lambda row: float(sum(row))/max(len(row),1))

    df['URL_Length'] = df.url.apply(lambda row: len(row))
    df['Query'] = df.url.apply(lambda row: urlparse.urlparse(row).query if urlparse.urlparse(row).query != '' else 'None')
    df['Query_Length'] = df.Query.apply(lambda row: len(row))
    df['Primary'] = df[domain_list_col].apply(lambda row: row[1] if row[0] in ['www','ww1','ww2'] else row[0])
    df['Primary_Length'] = df.Primary.apply(lambda row: len(row))
    df['Num_Periods'] = df.url.apply(lambda row: row.count("."))
    df['Num_Quest'] = df.url.apply(lambda row: row.count("?"))
    df['Num_Perc'] = df.url.apply(lambda row: row.count("%"))
    df['Num_Exclam'] = df.url.apply(lambda row: row.count("!"))
    df['Num_Numbers'] = df.url.apply(lambda row: sum(c.isdigit() for c in row))
    df['Words'] = df.url.apply(lambda row: "".join((char if char.isalnum() else " ") for char in row).split())
    df['Word_Counts'] = df.Words.apply(lambda row: len([len(x) for x in row]))
  #  df['Word_Counts'] = df.Word_Counts(lambda row: np.count_nonzero(row))


# # 3) Final creation of DataFrame and Export

# In[19]:


urlVars(df,'domain_temp')
df.drop(['Primary','Words','Query'],axis=1,inplace=True)


# In[ ]:


df.to_json('serving_input_data.json')

