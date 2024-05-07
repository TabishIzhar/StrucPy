
#Example 7: Analysis of 5-story irregular building with floor loads and self weight for multiple load combinations using :class:`StrucPy.RCFA.RCFenv`. Floor Loads generated using method autoflooring, and changes are made using inbuilt method.

from StrucPy.RCFA import RCFenv
import pandas as pd
import ray

# Importing Input Data from Excel File (Note: Change the Path as per the location of File)
member_details= pd.read_excel('./InputFiles/Testfile7.xlsx', 'members', header = 0, index_col=0)
nodes_details= pd.read_excel('./InputFiles/Testfile7.xlsx', 'nodes', header = 0, index_col=0)
boundcond = pd.read_excel('./InputFiles/Testfile7.xlsx', 'boundary', header = 0, index_col=0)
load_combo= pd.read_excel('./InputFiles/Testfile7.xlsx', 'load_combinations', header = 0, index_col=0)
seismic_defination= pd.read_excel('./InputFiles/Testfile7.xlsx', 'Seismic_Defination', header = 0, index_col=0)


# Creating RCFenv object "r1" for structure analysis
r1= RCFenv(nodes_details,member_details,boundcond, load_combo= load_combo, autoflooring= True, seismic_def= seismic_defination)

#Pre processing the model
r1.preP()

# Changing all floor loads with Floor Finish load as 50kN/m2
r1.changeFL(thickness= 0, LL=-25, FF=-50)

# Performing Analysis
r1.RCanalysis()

# Getting Reactions for every load combinations at base nodes 
base_reactions= r1.getReact()

# Getting Maximum Reactions at base nodes
base_max_reactions= r1.getReactmax()

# Getting nodal displacement for every load combinations at all nodes
nodal_displacements= r1.getNdis()

# Getting max nodal displacement at all nodes
nodal_max_displacements= r1.getNdismax()

# Getting member forces at extreme end of all members for every load combinations
end_member_forces= r1.getEndMF()

# Getting maximum positive and negative forces in all members for every load combinations
end_member_forces_max= r1.getMFmax()

# Getting maximum (design) forces in all members 
design_member_forces= r1.getMFdsg()

# Getting members displacement data for every load combinations in local coordinate system.
member_displacements_local= r1.getLDef()

# Getting members displacement data for every load combinations in global coordinate system.
member_displacements_global= r1.getGDef()


# Getting maximum displacement data for members in local coordinate system.
member_displacements_local= r1.getLDefmax()

# Getting maximum displacement data for members in global coordinate system.
member_displacements_global= r1.getGDefmax()

# Retrieving all the objects corresponding to different load combinations
LCobj= r1.getTLC()


#All the method of class RCF can be accessed for all the objects corresponding to different load combinations

LC1= LCobj[0]
LC2= LCobj[1]
LC3= LCobj[2]
LC4= LCobj[3]    #..so no upto LC11= LCobj[10]

#Accessing 3D model, 3D model will be same for every objects
model_3D = ray.get(LC1.model3D.remote())
model_3D.show()

#Accessing deflected 3D shape of model for LC1 and LC4
def_model_3D_LC1 = ray.get(LC1.def3D.remote())
def_model_3D_LC4 = ray.get(LC4.def3D.remote())

def_model_3D_LC1.show()
def_model_3D_LC4.show()
