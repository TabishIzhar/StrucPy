from StrucPy.RCFA import RCF
from StrucPy.RCFA import RCFenv
import pandas as pd
import ray

# importing modeling and data information from excel in dataframe
framegen=  pd.read_excel('./InputFiles/Example1.xlsx', 'framegen', header = 0, index_col=0)
seismic_defination= pd.read_excel('./InputFiles/Example2.xlsx', 'Seismic_Defination', header = 0, index_col=0)
load_combos= pd.read_excel('./InputFiles/Example2.xlsx', 'load_combinations_2', header = 0, index_col=0)

# Creating RC frame object for developing model
r1= RCF(nodes_details = None,member_details= None,boundarycondition= None,framegen= framegen, autoflooring= True)

r1.preP()

mem_list = [121,122,123,125,126,127,129,130,131,111,112,113,114,115,116,138,144,150,156,162,168,174,180,186,99,100,103,104,107,108,89,90,91,92,137,143,149,155,161,167,67,68,77,81,85,136,148,142]
r1.changeFrame(member= mem_list, delete= True)

r1.changeFrame(member= 'all',width= 300, depth= 500)
r1.changeFrame(member= 'beam', yudl= -10)
r1.preP()
r1.changeFL(thickness= 100, LL=0, FF=-10, WP=0)

member_details= r1.modelMD()
nodes_details= r1.modelND()
boundary_conditions= r1.modelBCD()
floor_detail= r1.floorD()
 
member_details.drop(['Type'], axis=1,inplace=True)
floor_detail.drop(['Floor'], axis=1,inplace=True)

# Creating RC frame object for structural anlaysis for different load combinations
r2= RCFenv(nodes_details = nodes_details, member_details= member_details, boundarycondition= boundary_conditions , load_combo= load_combos, seismic_def= seismic_defination, slab_details= floor_detail)


r2.preP()


r2.RCanalysis()

# Generates envelop for nodal displacement
getNdis= r2.getNdis()

# Generates envelop for maximum values of reactions
getReactmax= r2.getReactmax()

# Generates envelop for reactions from every load combinations
getReact= r2.getReact()

# Generates envelop for maximum values of nodal displacements
getNdismax= r2.getNdismax()

# Generates envelop for nodal forces in all members
getEndMF= r2.getEndMF()

# Generates envelop for all member forces that are to be used in designing.
getMFdsg= r2.getMFdsg()

# Getting ray actors for each load combinations
obj= r2.getTLC()

# Accessing seismic detail of load combination LC6
LC6_seismicD= ray.get(obj[7].seismicD.remote())

# Accessing story drift of load combination LC7
LC7_seismicD= ray.get(obj[8].seismicD.remote())

with pd.ExcelWriter('output_EXAMPLE2.xlsx') as writer:  
    getNdis.to_excel(writer, sheet_name='Nodal DISP ALL')
    getReact.to_excel(writer, sheet_name='All reactions')
    getNdismax.to_excel(writer, sheet_name='Nodal disp')
    getEndMF.to_excel(writer, sheet_name='End Moments')
    getMFdsg.to_excel(writer, sheet_name='Member Design Forces')
















