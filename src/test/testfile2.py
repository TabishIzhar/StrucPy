#Example 1: Analysis of fixed Beam With multiple point loads using :class:`StrucPy.RCFA.RCF`.

from StrucPy.RCFA import RCF
import pandas as pd
import numpy as np
import plotly

# Importing Input Data from Excel File (Note: Change the Path as per the location of File)
member_details= pd.read_excel('Testfiles/Testfile2_inputfile.xlsx', 'members', header = 0, index_col=0)
nodes_details= pd.read_excel('Testfiles/Testfile2_inputfile.xlsx', 'nodes', header = 0, index_col=0)
boundcond = pd.read_excel('Testfiles/Testfile2_inputfile.xlsx', 'boundary', header = 0, index_col=0)
forcevec= pd.read_excel('Testfiles/Testfile2_inputfile.xlsx', 'forcevec', header = 0, index_col=0)

actual_reactions1= np.array([[0,26.25,0,0,0,17.5],
                             [0,26.25,0,0,0,-17.5]])

actual_disp1= np.array([[0,0,0,0,0,0],
                        [0,0,0,0,0,0]])

actual_max_positive_BM=  26.25
actual_max_negative_BM=  -26.25
actual_max_positive_SF=  17.5
actual_max_negative_SF=  -8.75

# Self weight is being ignored
r1= RCF(nodes_details,member_details,boundarycondition= boundcond, forcesnodal=forcevec, self_weight=False) 

#Pre processing the model
r1.preP()

# Performing Analysis
r1.RCanalysis()

# Getting Reactions
reactions= r1.reactions()

# Getting Displacement
node_disp= r1.Gdisp()

# Getting Global stiffness matrix
Global_stiffness= r1.GlobalK()
detGK = np.linalg.det(Global_stiffness) 

# # Getting Local stiffness matrix
# Local_stiffness= r1.LocalK()
# detLK = np.linalg.det(Local_stiffness) 

# # Getting Shear Force and Bending Moment Diagram of Member with ID- 1
# sfbmd= r1.sfbmd(1)
# memF= r1.memF()    
# gen_max_positive_BM=  np.max(memF[0][:,5])
# gen_max_negative_BM=  np.min(memF[0][:,5])
# gen_max_positive_SF=  np.max(memF[0][:,2])
# gen_max_negative_SF=  np.min(memF[0][:,2])

# # Getting Deflection of Member with ID- 1
# defLD= r1.defLD(1)
# defGD= r1.defGD()
# member_disp_local= defLD[0]   
# member_disp_global= defGD[0] 
# if  abs(np.max(member_disp_local[:,2]))>= abs(np.min(member_disp_local[:,2])):
#     gen_max_disp_local=  np.max(member_disp_local[:,2])
# else:
#     gen_max_disp_local=  np.min(member_disp_local[:,2])

# if  abs(np.max(member_disp_global[:,2]))>= abs(np.min(member_disp_global[:,2])):
#     gen_max_disp_global=  np.max(member_disp_global[:,2])
# else:
#     gen_max_disp_global=  np.min(member_disp_global[:,2])


