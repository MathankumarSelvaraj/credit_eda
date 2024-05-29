
#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
#ignoring warnings
import warnings
warnings.filterwarnings("ignore")
#setting the dark of theme for graphics
sns.set_style("dark")
from scipy import stats


# Problem Statement: Understand how the bank approves and refuses loan. Find out different patterns and represent the outcomes to help the bank reduce the credit risk and interest risk.
# 
# The two input files are extracted, cleaned/transformed and few columns are analyzed via different charts generated using different Python libraries. Then some inferences are made based on the outcomes.

# ## 1) Application Data: Data Exploration


#reading application data from local disc
application_data = pd.read_csv(r'D:\Data_Science\UpGrad\Case Study\application_data.csv')


# In[3]:


application_data.head()


# In[4]:


#shape of no of records
print("No of Records: ",application_data.shape[0])
print("No of features: ",application_data.shape[1])


# In[5]:


# number of missing values and percentage of the same
missing_values = pd.DataFrame(application_data.isnull().sum().rename("cnt_missing_values"))
missing_values["in_percentage"] = (round(missing_values["cnt_missing_values"]/application_data.shape[0],2))*100


# In[6]:


#features which has 14% and more than 14% missing values
missing_values[missing_values["in_percentage"]>=0.14]


#  
# 1) There are some meaning in missing values, for example "OWN_CAR_AGE" has 66% of missing values because those appicants don't have a car.
# 
# 2) Since I'am not doing data preparation for ML model. So, I'm not removing missing values attributes to explore the data. However, I have done some missing value treatment.

# In[7]:


#Type of each variable
application_data.info(verbose= True)


# In[8]:


# Checking count of defaulters and non-defaulters
application_data["TARGET"].value_counts()


# In[9]:


#I decided below columns are not important of my initial analysis
unwanted_cols = ["FLAG_DOCUMENT_2","FLAG_DOCUMENT_3","FLAG_DOCUMENT_4","FLAG_DOCUMENT_5",
                     "FLAG_DOCUMENT_6","FLAG_DOCUMENT_7","FLAG_DOCUMENT_8","FLAG_DOCUMENT_9",
                     "FLAG_DOCUMENT_10","FLAG_DOCUMENT_11","FLAG_DOCUMENT_12","FLAG_DOCUMENT_13",
                     "FLAG_DOCUMENT_14","FLAG_DOCUMENT_15","FLAG_DOCUMENT_16","FLAG_DOCUMENT_17",
                     "FLAG_DOCUMENT_18","FLAG_DOCUMENT_19","FLAG_DOCUMENT_20","FLAG_DOCUMENT_21",
                    "EXT_SOURCE_1","EXT_SOURCE_3"]


# In[10]:


# removing unwanted columns 
i=0
while i< len(unwanted_cols):
    del application_data[unwanted_cols[i]]
    i+=1


# In[11]:


len(application_data.columns)


# In[12]:


#taking subset of application data for shake of analysis
app_sub1 = application_data[["SK_ID_CURR","TARGET","NAME_CONTRACT_TYPE","CODE_GENDER","FLAG_OWN_CAR","FLAG_OWN_REALTY",
                            "CNT_CHILDREN","AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","AMT_GOODS_PRICE",
                            "NAME_INCOME_TYPE","NAME_EDUCATION_TYPE","NAME_FAMILY_STATUS","NAME_HOUSING_TYPE",
                            "DAYS_BIRTH","DAYS_EMPLOYED","OWN_CAR_AGE","CNT_FAM_MEMBERS","ORGANIZATION_TYPE"]]


# In[13]:


app_sub1.head()


# In[14]:


# Getting the counts of number of people owning a car.
app_sub1.FLAG_OWN_CAR.value_counts()


# In[ ]:


#Finding a new app_sub1.isnull().sum()


# In[16]:


app_sub1["FLAG_OWN_CAR"].value_counts()


# In[17]:


app_sub1["OWN_CAR_AGE"]


# In[18]:


app_sub1.loc[(app_sub1["OWN_CAR_AGE"].isnull()) & (app_sub1["FLAG_OWN_CAR"]=="N"),'OWN_CAR_AGE']='NotApplicable'


# In[19]:


app_sub1.isnull().sum()


# In[20]:


app_sub1['AMT_GOODS_PRICE'].fillna(np.mean(app_sub1["AMT_GOODS_PRICE"]), inplace=True)
app_sub1['AMT_ANNUITY'].fillna(np.mean(app_sub1["AMT_ANNUITY"]), inplace=True)
app_sub1['OWN_CAR_AGE'].fillna(np.mean(app_sub1[app_sub1["OWN_CAR_AGE"]!="NotApplicable"].OWN_CAR_AGE), inplace=True)


# In[21]:


app_sub1.dropna(inplace=True)


# In[22]:


app_sub1.isnull().sum()


# In[23]:


app_sub1.head(3)


# In[24]:


ax = sns.countplot(app_sub1["TARGET"])

for p in ax.patches:
    ax.annotate('{:.1f}%'.format((p.get_height()/len(app_sub1))*100), (p.get_x()+0.3, p.get_height()+50),
                fontsize=13, color='g',ha='center', va='bottom')

plt.title("PROPORTION OF DEFAULTERS",fontsize=18)
plt.xlabel('Count of Applications',fontsize=15)
plt.ylabel('Flag of Defaulter',fontsize=15)
plt.show()
plt.savefig("output.jpg")
plt.savefig("output1", facecolor='g', bbox_inches="tight",
            pad_inches=0.3, transparent=True)


# 
# In the application data, the proportion of defaulters and non-defaluters is significatly different from each other. In other words, the data is imblanaced

# In[25]:




Gender = app_sub1[app_sub1["TARGET"]==0].CODE_GENDER.value_counts().index[:2]
non_defaulters = app_sub1[app_sub1["TARGET"]==0].CODE_GENDER.value_counts().values[:2]
defaulters = app_sub1[app_sub1["TARGET"]==1].CODE_GENDER.value_counts().values[:2]
# Make figure and axes
colors1 = ['#99ff99','#ffcc99']
fig, axs = plt.subplots(1, 2,figsize=(10, 10),dpi=80, facecolor='w', edgecolor='k')

# A standard pie plot
axs[1].pie(non_defaulters, labels=Gender, autopct='%1.1f%%', shadow=True,colors=colors1)
colors2 = ['#ff9999','#66b3ff']
# Shift the second slice using explode
axs[0].pie(defaulters, labels=Gender, autopct='%1.1f%%', shadow=True,
              explode=(0, 0),colors=colors2)
#axs[0].title("Defaulter")

axs[1].title.set_text('Gender Distribution for Non-Defaulters')
axs[0].title.set_text('Gender Distribution for Defaulters')

#
plt.show()


# In both case, Female applicants are higher than male applicants.

# In[26]:


table = pd.pivot_table(app_sub1, values="NAME_CONTRACT_TYPE",columns="TARGET",index="CODE_GENDER",aggfunc="count")


# In[27]:


table


# In[28]:


#Distribution plots as a function

def vbar_distplot(variable):
    plt.figure(figsize=(15,5))
    plt.subplot(1, 2, 1)
    ax1=sns.countplot(variable, data=app_sub1[app_sub1["TARGET"]==0],alpha=0.5)
    plt.title('Distribution of '+ '%s' %variable +' for Non-Defaulters', fontsize=14)
    for p in ax1.patches:
        ax1.annotate('{:.1f}%'.format((p.get_height()/len(app_sub1[app_sub1["TARGET"]==0]))*100), (p.get_x()+0.3, p.get_height()+50),
                fontsize=13, color='g',ha='center', va='bottom')


    plt.xlabel(variable)
    plt.xticks(rotation=0)
    plt.ylabel('Number of cases for non-defaulters')
    plt.subplot(1, 2, 2)
    ax2=sns.countplot(variable, data=app_sub1[app_sub1["TARGET"]==1],alpha=0.5)
    for p in ax2.patches:
        ax2.annotate('{:.1f}%'.format((p.get_height()/len(app_sub1[app_sub1["TARGET"]==1]))*100), (p.get_x()+0.3, p.get_height()+50),
                fontsize=13, color='g',ha='center', va='bottom')
    plt.title('Distribution of '+ '%s' %variable +' for Defaulters', fontsize=14)
    plt.xlabel(variable)
    plt.xticks(rotation=0)
    plt.ylabel('Number of cases for defaulters')
    plt.show()


# #### Count of how many people having a car

# In[29]:


vbar_distplot("FLAG_OWN_CAR")


# In[30]:


vbar_distplot("FLAG_OWN_REALTY")


# In[31]:


vbar_distplot("NAME_CONTRACT_TYPE")


# In[32]:


def hbar_distplot(variable):
    plt.figure(figsize=(10,12))
    plt.subplot(2, 1, 1)
    ax1=sns.countplot(y=variable, data=app_sub1[app_sub1["TARGET"]==0],alpha=0.7)
    plt.title('Distribution of '+ '%s' %variable +' for Non-Defaulters', fontsize=14)
    for p in ax1.patches:
        ax1.text(p.get_width(),p.get_y() + p.get_height()/2, '{:.1f}%'.format(round(float(p.get_width()/len(app_sub1[app_sub1["TARGET"]==0]))*100)), 
            fontsize=12, color='red', ha='left', va='center')

    plt.xlabel(variable)
    plt.xticks(rotation=0)
    plt.ylabel('Number of cases for non-defaulters')
    plt.subplot(2, 1, 2)
    ax2=sns.countplot(y=variable, data=app_sub1[app_sub1["TARGET"]==1],alpha=0.7)
    for p in ax2.patches:
        ax2.text(p.get_width(),p.get_y() + p.get_height()/2, '{:.1f}%'.format(round(float(p.get_width()/len(app_sub1[app_sub1["TARGET"]==1]))*100)), 
            fontsize=12, color='red', ha='left', va='center')
    plt.title('Distribution of '+ '%s' %variable +' for Defaulters', fontsize=14)
    plt.xlabel(variable)
    plt.xticks(rotation=0)
    plt.ylabel('Number of cases for defaulters')
    plt.show()


# In[33]:


hbar_distplot("NAME_INCOME_TYPE")


# In[34]:


hbar_distplot("NAME_FAMILY_STATUS")


# In[35]:


app_sub1['AGE'] =app_sub1['DAYS_BIRTH']//-365.25
app_sub1.drop(['DAYS_BIRTH'],axis=1,inplace=True)
app_sub1['AGE_GROUP']= pd.cut(app_sub1.AGE,bins=np.linspace(20 ,70,num=11))


# In[36]:


app_sub1.AGE_GROUP.value_counts()


# In[37]:


hbar_distplot("AGE_GROUP")


# In[38]:


temp = app_sub1[app_sub1["OWN_CAR_AGE"]!="NotApplicable"]
app_car_age=temp[temp.OWN_CAR_AGE<np.percentile(temp['OWN_CAR_AGE'], 99)]


# In[39]:


plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
ax1=sns.distplot(app_car_age[app_car_age["TARGET"]==0].OWN_CAR_AGE)
plt.title('Distribution of car age for Non-Defaulters', fontsize=14)


plt.xlabel("Car age")
plt.xticks(rotation=0)
plt.ylabel('Number of cases for non-defaulters')
plt.subplot(1, 2, 2)
ax2=sns.distplot(app_car_age[app_car_age["TARGET"]==1].OWN_CAR_AGE)

plt.title('Distribution of car age for Defaulters', fontsize=14)
plt.xlabel("Car age")
plt.xticks(rotation=0)
plt.ylabel('Number of cases for defaulters')
plt.show()


# In[40]:


vbar_distplot("CNT_FAM_MEMBERS")


# In[41]:


app_sub1['AMT_INCOME_TOTAL'].describe()


# In[42]:


plt.figure(figsize=(10,8))
plt.subplot(2, 1, 1)
ax1=sns.boxplot(app_sub1[app_sub1["TARGET"]==0].AMT_INCOME_TOTAL)
plt.title('Distribution of income for Non-Defaulters', fontsize=10)
plt.xlabel("Income")

plt.ylabel('non-defaulters')
plt.subplot(2, 1, 2)
ax1=sns.boxplot(app_sub1[app_sub1["TARGET"]==1].AMT_INCOME_TOTAL)
plt.title('Distribution of income for Defaulters', fontsize=10)
plt.xlabel("Income")

plt.ylabel('defaulters')


# In[43]:


app_sub1=app_sub1[app_sub1.AMT_INCOME_TOTAL<np.nanpercentile(app_sub1['AMT_INCOME_TOTAL'], 99)]


# In[44]:


plt.figure(figsize=(10,8))
plt.subplot(2, 1, 1)
ax1=sns.boxplot(app_sub1[app_sub1["TARGET"]==0].AMT_INCOME_TOTAL)
plt.title('Distribution of income for Non-Defaulters', fontsize=10)
plt.xlabel("Income")

plt.ylabel('non-defaulters')
plt.subplot(2, 1, 2)
ax1=sns.boxplot(app_sub1[app_sub1["TARGET"]==1].AMT_INCOME_TOTAL)
plt.title('Distribution of income for Defaulters', fontsize=10)
plt.xlabel("Income")

plt.ylabel('defaulters')


# In[45]:


#Creating binned var
app_sub1.loc[:,'INCOME_RANGE']=pd.qcut(app_sub1.loc[:,'AMT_INCOME_TOTAL'],q=[0,0.20,0.50,0.90,1],
labels=['Low','Medium','High','Very_high'])


# In[46]:


vbar_distplot("INCOME_RANGE")


# In[47]:


app_sub1['YEARS_EMPLOYED'] =app_sub1['DAYS_EMPLOYED']//-365.25


# In[48]:


hbar_distplot("YEARS_EMPLOYED")


# In[49]:


app_sub1[app_sub1['DAYS_EMPLOYED']//-365.25<0]


# In[50]:


plt.figure(figsize=(10,8))
plt.subplot(2, 2, 1)
sns.distplot(app_sub1["AMT_CREDIT"])
plt.subplot(2, 2, 2)
ax2=sns.distplot(app_sub1["AMT_ANNUITY"],color="g")
plt.subplot(2, 2, 3)
ax2=sns.distplot(app_sub1["AMT_GOODS_PRICE"],color="y")
plt.subplot(2, 2, 4)
ax2=sns.distplot(app_sub1["AMT_INCOME_TOTAL"],color="r")


# In[51]:


plt.figure(figsize=(18,6))
plt.subplot(121)
sns.scatterplot(x='AMT_CREDIT',y='AMT_INCOME_TOTAL',data=app_sub1[app_sub1["TARGET"]==0])
plt.title('INCOME vs CREDIT for Non-Defaulters')

plt.subplot(122)
sns.scatterplot(x='AMT_CREDIT',y='AMT_INCOME_TOTAL',data=app_sub1[app_sub1["TARGET"]==0],color="r")
plt.title('INCOME vs CREDIT for Defaulters')
plt.show()


# In[52]:


plt.figure(figsize=(18,6))
plt.subplot(121)
sns.scatterplot(x='AMT_CREDIT',y='AMT_GOODS_PRICE',data=app_sub1[app_sub1["TARGET"]==0],color="g")
plt.title('CREDIT vs GOODS PRICE for Non-Defaulters')

plt.subplot(122)
sns.scatterplot(x='AMT_CREDIT',y='AMT_GOODS_PRICE',data=app_sub1[app_sub1["TARGET"]==0],color="r")
plt.title('CREDIT vs GOODS PRICE for Defaulters')
plt.show()


# In[53]:


numerical_ft=list(app_sub1.select_dtypes('float').columns)
d=app_sub1[numerical_ft]
d_corr = d.corr()
round(d_corr, 3)


# In[54]:


sns.heatmap(d.corr(), cmap="YlGnBu", annot=True)


# In[55]:


previous_app = pd.read_csv(r'D:\Data_Science\UpGrad\Case Study\previous_application.csv')


# In[56]:


previous_app.head()


# In[57]:


print("No of Records: ",previous_app.shape[0])
print("No of features: ",previous_app.shape[1])


# In[58]:


previous_app.info()


# In[65]:


previous_app.dtypes.value_counts()


# In[66]:


missing_values_pre = pd.DataFrame(previous_app.isnull().sum().rename("cnt_missing_values"))
missing_values_pre["in_percentage"] = (round(missing_values_pre["cnt_missing_values"]/previous_app.shape[0],2))*100


# In[67]:


missing_values_pre[missing_values_pre["in_percentage"]>=25.0]


# In[62]:


for x in missing_values_pre[missing_values_pre["in_percentage"]>=25.0].index:
    del previous_app[x]


# In[63]:


previous_app.shape


# In[64]:


def plot_by_cat_num(cat, num):

    plt.style.use('ggplot')
    sns.despine
    fig,ax = plt.subplots(1,1,figsize=(10,8))
    
    sns.boxenplot(x=cat,y = num, data=previous_app)
    ax.set_ylabel(f'{num}')
    ax.set_xlabel(f'{cat}')

    ax.set_title(f'{cat} Vs {num}',fontsize=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
     
    plt.show()


# In[68]:


plot_by_cat_num('NAME_CONTRACT_STATUS', 'AMT_ANNUITY')


# In[69]:


plot_by_cat_num('NAME_CONTRACT_STATUS', 'AMT_CREDIT')


# In[70]:


combined = pd.merge(app_sub1, previous_app, how='inner', on=['SK_ID_CURR'])


# In[71]:


plt.figure(figsize=(7,5))
ax =sns.countplot(previous_app.NAME_CONTRACT_STATUS,alpha=0.7)
for p in ax.patches:
    ax.annotate('{:.1f}%'.format((p.get_height()/len(previous_app))*100), (p.get_x()+0.2, p.get_height()+50),
                fontsize=13, color='g',ha='left', va='bottom')
plt.ylabel("Count of Contract Status")
plt.title("Distribution of Contract Status")
plt.show()


# In[72]:


combined.shape


# In[73]:


approved=combined[combined.NAME_CONTRACT_STATUS=='Approved']
refused=combined[combined.NAME_CONTRACT_STATUS=='Refused']
canceled=combined[combined.NAME_CONTRACT_STATUS=='Canceled']
unused=combined[combined.NAME_CONTRACT_STATUS=='Unused Offer']


# In[74]:


def plots_app(var):
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15,5))
    
    s1=sns.countplot(ax=ax1,x=refused[var], data=refused, order= refused[var].value_counts().index,alpha=0.7)
    for p in ax1.patches:
        ax1.text(p.get_x() + p.get_width()/2., p.get_height(),"{:.1f}%".format(round((p.get_height()/len(refused))*100)), 
            fontsize=12, color='g', ha='center', va='bottom')
    ax1.set_title("Refused", fontsize=10)
    ax1.set_xlabel('%s' %var)
    ax1.set_ylabel("Count of Loans")
    s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
    
    s2=sns.countplot(ax=ax2,x=approved[var], data=approved, order= approved[var].value_counts().index,alpha=0.7)
    for p in ax2.patches:
        ax2.text(p.get_x() + p.get_width()/2., p.get_height(),"{:.1f}%".format(round((p.get_height()/len(approved))*100)), 
            fontsize=12, color='g', ha='center', va='bottom')
    s2.set_xticklabels(s2.get_xticklabels(),rotation=90)
    ax2.set_xlabel('%s' %var)
    ax2.set_ylabel("Count of Loans")
    ax2.set_title("Approved", fontsize=10)
    
    
    s3=sns.countplot(ax=ax3,x=canceled[var], data=canceled, order= canceled[var].value_counts().index,alpha=0.7)
    for p in ax3.patches:
        ax3.text(p.get_x() + p.get_width()/2., p.get_height(),"{:.1f}%".format(round((p.get_height()/len(canceled))*100)), 
            fontsize=12, color='g', ha='center', va='bottom')
    ax3.set_title("Canceled", fontsize=10)
    ax3.set_xlabel('%s' %var)
    ax3.set_ylabel("Count of Loans")
    s3.set_xticklabels(s3.get_xticklabels(),rotation=90)
    plt.show()


# In[75]:


plots_app('TARGET')


# In[76]:


plots_app('CODE_GENDER')


# In[77]:


plots_app('FLAG_OWN_CAR')


# In[78]:


refused.columns


# In[79]:


plots_app('FLAG_OWN_REALTY')


# In[80]:


plots_app('NAME_INCOME_TYPE')


# In[81]:


plots_app('AGE_GROUP')


# In[82]:


plots_app('INCOME_RANGE')


# In[83]:


plt.figure(figsize=(20,6))
plt.subplot(131)
sns.scatterplot(x='AMT_CREDIT_x',y='AMT_INCOME_TOTAL',data=approved)
plt.title('INCOME vs CREDIT for Non-Defaulters')

plt.subplot(132)
sns.scatterplot(x='AMT_CREDIT_x',y='AMT_INCOME_TOTAL',data=refused,color="y")
plt.title('INCOME vs CREDIT for Defaulters')
plt.show()

