#Example 1: Analysis of fixed Beam With multiple point loads using :class:`StrucPy.RCFA.RCF`.

from StrucPy.RCFA import RCF
import pandas as pd
import plotly
import time

start = time.time()

framegen=  pd.read_excel('./InputFiles/Example1.xlsx', 'framegen', header = 0, index_col=0)
seismic_defination= pd.read_excel('./InputFiles/Example1.xlsx', 'Seismic_Defination', header = 0, index_col=0)
load_combo= pd.read_excel('./InputFiles/Example1.xlsx', 'load_combinations', header = 0, index_col=0)

# Creating RC frame object for analysis
r1= RCF(nodes_details = None,member_details= None,boundarycondition= None,framegen= framegen, seismic_def= seismic_defination, load_combo= load_combo,  autoflooring= True)

#Pre processing the model
r1.preP()

"""If using VS Code or JupiterNotebook: models can be viewed by `model.show()` otherwise save model as html `model.write_html("./name.html")`
"""
# Initial model
model = r1.model3D()
model.write_html("./model3D.html")

# Deleting few members 
mem_list = [121,122,123,125,126,127,129,130,131,111,112,113,114,115,116,138,144,150,156,162,168,174,180,186,99,100,103,104,107,108,89,90,91,92,137,143,149,155,161,167,67,68,77,81,85,136,148,142]
r1.changeFrame(member= mem_list, delete= True)

#Changing width and depth of all members to 300 X 500 and udl of -10kN/m in y-direction on all members
r1.changeFrame(member= 'all',width= 300, depth= 500, yudl= -10)

# New edited model
modelN = r1.model3D()
modelN.write_html("./model3DN.html")

#Changing the floor thickness and loads 
r1.changeFL(thickness= 100, LL=0, FF=-10, WP=0)

# Performing Analysis
r1.RCanalysis()


end2 = time.time()
print(end2 - end1)

# # Getting Reactions
# reactions= r1.reactions()

# # Getting Shear Force and Bending Moment Diagram of Member with ID- 1
# sfbmd= r1.sfbmd(1)
# # sfbmd.show()             # If using VS Code or JupiterNotebook


# # Getting Material Properties
# material_properties= r1.Mproperties()

# end = time.time()
# print(end - start)