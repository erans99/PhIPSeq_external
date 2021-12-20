import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from config import base_path
from config import out_path

#this script creates the dicriptive figure #1, because of design and stalying prefrences the boxplot, bar chart and
#pie chart was done in also excel using the csv created in the script, and this is what was shown in the paper. the histogram was done in python

#part 1 outputs the boxplot of how many oligos passing per person in the cohort

MIN_OLIS = 200
THROW_BAD_OLIS = True
df_info = pd.read_csv(os.path.join(base_path, "library_contents.csv"), index_col=0, low_memory=False)
#df_info = df_info[(df_info.is_allergens | df_info.is_IEDB) & (df_info['num_copy'] == 1)]
inds = df_info.index
meta_df = pd.read_csv(os.path.join(base_path, "cohort.csv"), index_col=0, low_memory=False)
# if you want the base samples
meta_df = meta_df[(meta_df.timepoint == 1) & (meta_df.num_passed >= MIN_OLIS)]

#fold and exist
fold_df = pd.read_csv(os.path.join(base_path, "fold_data.csv"), index_col=[0, 1],
                          low_memory=False).loc[meta_df.index].unstack()

#fold_df.columns = fold_df.columns.get_level_values(1)
#fold_df = fold_df[fold_df.columns.intersection(inds)]

if THROW_BAD_OLIS:
    drop = fold_df.columns[(fold_df == -1).sum() > 0]
    fold_df = fold_df[fold_df.columns.difference(drop)].fillna(1)
else:
    fold_df = fold_df.fillna(1)

df_exist = (fold_df > 1).astype(int).T

list_of_dfs = [df_info,meta_df, fold_df, df_exist]

def shape_of_df(data):
    name =[x for x in globals() if globals()[x] is data][0]
    print("Dataframe Name is: %s" % name)
    print(data.shape)
    print(data.head())
for a in list_of_dfs:
    shape_of_df(a)

absol_passed = pd.DataFrame(df_exist.sum(0))
absol_passed.columns = ['abs_passed']
absol_passed.sort_values(by='abs_passed', ascending=True, inplace=True)
absol_passed.columns = ['']

boxplot = absol_passed.boxplot(grid = False, figsize = (3,7))
plt.ylabel("Number of significantly bound peptides per individual")
plt.xlabel("")
#plt.show()
fig_file_name = 'twist_num_peptides_passed_per_serum.png'
plt.savefig(os.path.join(out_path, fig_file_name))
plt.clf()
absol_passed.to_csv(os.path.join(out_path, 'num_peptides_passed_per_serum_cleandata.csv'))

#part 2 outputs the bar chart of how many peptides pass in how many indeviduals by the groups

pep_passed = pd.DataFrame(df_exist.sum(1))
pep_passed.columns = ['num_serum_passed']
pep_passed.sort_values(by='num_serum_passed', ascending=True, inplace=True)

pep_passed.reset_index(inplace=True)
pep_passed.set_index('level_1',inplace=True)
pep_passed.drop(columns=['level_0'], inplace=True)

df_plot= pd.merge(df_info,pep_passed, how = 'left',left_index=True, right_index=True)
#fill all the nans that didnt pass at anyone with 0
df_plot['num_serum_passed'] = df_plot['num_serum_passed'].fillna(0)

#creating the diffrent catagories
def antigen_type(row):
    if row["is_all_cntrl"]==True and row["is_IEDB"]==False:
        return "control"
    elif row["is_infect"]==True and row["is_auto"]==False and row["is_all_cntrl"]==False and row["is_IEDB"]==True:
        return "IEBD infections disease"
    elif row["is_auto"]==True and row["is_infect"]==False and row["is_all_cntrl"]==False and row["is_IEDB"]==True:
        return "IEBD autoimmune diseases"
    #this line is added because I saw few rows with is control True but they are auto... example: twist_26306
    elif row["is_auto"] == True and row["is_infect"] == False and row["is_IEDB"] == True:
        return "IEBD autoimmune diseases"
    #this line is added because I saw few rows with is control True but they are infect... example: twist_26452
    elif row["is_auto"] == False and row["is_infect"] == True and row["is_IEDB"] == True:
        return "IEBD autoimmune diseases"
    elif row["is_auto"]==True and row["is_infect"]==True and row["is_IEDB"]==True:
        return "IEBD other"
    elif row["is_auto"]==False and row["is_infect"]==False and row["is_IEDB"]==True:
        return "IEBD other"
    elif row["is_animal"]== True:
        return "Allergens (Animal)"
    elif row["is_bacteria"]== True:
        return "Allergens (Bacterial)"
    elif row["is_fungi"]== True:
        return "Allergens (Fungal)"
    elif row["is_human"]== True:
        return "Allergens (Human)"
    elif row["is_insect"]== True:
        return "Allergens (Insect)"
    elif row["is_plant"]== True:
        return "Allergens (plant)"
    else:
        return "other"

df_plot['antigen_type'] = df_plot.apply(antigen_type, axis=1)

#if we want not to include controls in the fig use the following line
df_plot = df_plot.loc[df_plot['antigen_type']!= "control"]

fig_dict = {}
antigens_type_list = ['Allergens (plant)', 'Allergens (Animal)', 'Allergens (Insect)',\
                      'Allergens (Fungal)', 'Allergens (Human)', 'Allergens (Bacterial)',\
                      'IEBD autoimmune diseases', 'IEBD infections disease', 'IEBD other']
df_sum = pd.DataFrame(columns=[], index=antigens_type_list)

#creating the df for the bar plot, each bar dicribes the diffrents parts of the library at a spesific precentaage of passing
list_of_per = [0,1,5,10,20,30]
for a in list_of_per:
    df = df_plot.loc[df_plot["num_serum_passed"]>(a*10)-1]
    antigens = antigens_type_list
    for b in antigens:
        antigen = b
        fig_dict[str(antigen)]= str((len(df.loc[df["antigen_type"]== b])/len(df))*100)
    df_dict = pd.DataFrame(fig_dict, index=[0]).T
    df_sum = pd.merge(df_sum, df_dict, how = 'left',left_index=True, right_index=True)

columns=['Input library', '>1%', '>5%', '>10%', '>20%', '>30%']
df_sum.columns = columns
df_sum=df_sum.T
df_sum=df_sum.astype(float)# enables numeric data if not it gives en error
df_sum.plot.bar(stacked=True)#, legend=None)#takes the legent out
plt.xticks(rotation=0)#rotate the words try 0 0r 90
plt.tight_layout()#makes the words not cut out
#plt.show()
plt.savefig(os.path.join(out_path,"library_contant_in_diffrent_cutoffs.png"))
plt.clf()
df_sum.to_csv(os.path.join(out_path, 'lib_contant_by_precent_pass_cleandata.csv'))

#part 3 outputs the pie chart of the input library

pep_passed = pd.DataFrame(df_exist.sum(1))
pep_passed.columns = ['num_serum_passed']
pep_passed.sort_values(by='num_serum_passed', ascending=True, inplace=True)

pep_passed.reset_index(inplace=True)
pep_passed.set_index('level_1',inplace=True)
pep_passed.drop(columns=['level_0'], inplace=True)

df_plot= pd.merge(df_info,pep_passed, how = 'left',left_index=True, right_index=True)
#fill all the nans that didnt pass at anyone with 0
df_plot['num_serum_passed'] = df_plot['num_serum_passed'].fillna(0)


def antigen_type(row):
    if row["is_all_cntrl"]==True and row["is_IEDB"]==False:
        return "control"
    elif row["is_infect"]==True and row["is_auto"]==False and row["is_all_cntrl"]==False and row["is_IEDB"]==True:
        return "IEBD infections disease"
    elif row["is_auto"]==True and row["is_infect"]==False and row["is_all_cntrl"]==False and row["is_IEDB"]==True:
        return "IEBD autoimmune diseases"
    #this line is added because I saw few rows with is control True but they are auto... example: twist_26306
    elif row["is_auto"] == True and row["is_infect"] == False and row["is_IEDB"] == True:
        return "IEBD autoimmune diseases"
    #this line is added because I saw few rows with is control True but they are infect... example: twist_26452
    elif row["is_auto"] == False and row["is_infect"] == True and row["is_IEDB"] == True:
        return "IEBD autoimmune diseases"
    elif row["is_auto"]==True and row["is_infect"]==True and row["is_IEDB"]==True:
        return "IEBD other"
    elif row["is_auto"]==False and row["is_infect"]==False and row["is_IEDB"]==True:
        return "IEBD other"
    elif row["is_animal"]== True:
        return "Allergens (Animal)"
    elif row["is_bacteria"]== True:
        return "Allergens (Bacterial)"
    elif row["is_fungi"]== True:
        return "Allergens (Fungal)"
    elif row["is_human"]== True:
        return "Allergens (Human)"
    elif row["is_insect"]== True:
        return "Allergens (Insect)"
    elif row["is_plant"]== True:
        return "Allergens (plant)"
    else:
        return "other"

df_plot['antigen_type'] = df_plot.apply(antigen_type, axis=1)

fig_dict = {}
antigens_type_list = ['Allergens (plant)', 'Allergens (Animal)', 'Allergens (Insect)',\
                      'Allergens (Fungal)', 'Allergens (Human)', 'Allergens (Bacterial)',\
                      'IEBD autoimmune diseases', 'IEBD infections disease', 'IEBD other', 'control']
df_sum = pd.DataFrame(columns=[], index=antigens_type_list)

list_of_per = [0,1,5,10,50,90]
for a in list_of_per:
    df = df_plot.loc[df_plot["num_serum_passed"]>(a*10)-1]
    antigens = antigens_type_list
    for b in antigens:
        antigen = b
        fig_dict[str(antigen)]= str((len(df.loc[df["antigen_type"]== b])/len(df))*100)
        #if you want absolute numbers and not % use the following line
        #fig_dict[str(antigen)] = str(len(df.loc[df["antigen_type"] == b]))
    df_dict = pd.DataFrame(fig_dict, index=[0]).T
    df_sum = pd.merge(df_sum, df_dict, how = 'left',left_index=True, right_index=True)
columns=['Input library', '>1%', '>5%', '>10%', '>50%', '>90%']
df_sum.columns = columns
#df_sum.sort_index(inplace=True)

df_sum=df_sum.astype(float)# enables numeric data if not it gives en error
input = pd.Series(df_sum['Input library'])
input.plot.pie(figsize=(6, 6))
#plt.show()
plt.savefig(os.path.join(out_path,"library_contant_pie_chart.png"))
plt.clf()
input.to_csv(os.path.join(out_path, 'input_lib_contant_to_pie_chart.csv'))

#part 4 prepares the histogram with 2 colors, allergens and iedb

pep_passed = pd.DataFrame(df_exist.sum(1))
pep_passed.columns = ['num_serum_passed']
pep_passed.sort_values(by='num_serum_passed', ascending=True, inplace=True)
pep_passed.reset_index(inplace=True)
pep_passed.set_index('level_1',inplace=True)
pep_passed.drop(columns=['level_0'], inplace=True)

IEDB_index=set(df_info.loc[(df_info["is_IEDB"]== True )].index)& set(pep_passed.index)
Allergens_index=set(df_info.loc[(df_info["is_allergens"]== True )].index)& set(pep_passed.index)


pep_passed_IEDB = pep_passed.loc[IEDB_index]
pep_passed_IEDB = pep_passed_IEDB['num_serum_passed']
pep_passed_Allergens = pep_passed.loc[Allergens_index]
pep_passed_Allergens= pep_passed_Allergens['num_serum_passed']

plt.hist([pep_passed_Allergens, pep_passed_IEDB], bins=50, label = ['allergens DBs','IEDB'])
plt.legend(loc='upper right')
plt.ylabel("Number of significantly bound peptides (log)")
plt.xlabel("Number of individuals")
plt.yscale('log')
plt.xticks(np.arange(0, 1100, step=100))
plt.title("")
plt.grid(False)
#plt.show()

fig_file_name = 'twist_hist_peptides_individuals_allergens_IEDB_cleandata_281021.png'
plt.savefig(os.path.join(out_path, fig_file_name))
plt.clf()
