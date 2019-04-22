import pandas as pd
import urllib.parse as urlparse

validExtensions = pd.read_excel("./Extensions.xlsx")
validExtensions = list(validExtensions['Ext'])


def primary_domain(x, l='len'):
    temp = urlparse.urlparse(x).hostname
    if l == 'len':
        return len(temp)
    else:
        return temp


def ext_find(url, combine=True):
    t = primary_domain(url, l='str')
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
    if combine is True:
        if orig[:2] == "ww":
            t = [t[0], ".".join(t[1:])]
        else:
            t = [".".join(t)]
    t.append(temp)
    # print(str(orig) + " --> " + str(t))
    return t


def process_date(df):
    """Only keep the data format as Mon, 21 Jan 2013 19:31:19 GMT. Other dates are set to "Unknown
    """
    df.lastModified = df.lastModified.apply(lambda row: 'Unknown' 
                                            if row[2:].isnumeric() else row)
    
    df['last_year_modified'] = df.lastModified.apply(lambda row: str(row.split(' ')[3]) 
                                                     if row != 'Unknown' else row) 
    return df['last_year_modified']
