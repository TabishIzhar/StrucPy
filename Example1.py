#Example 1: Analysis of fixed Beam With multiple point loads using :class:`StrucPy.RCFA.RCF`.

from StrucPy.RCFA import RCF
import pandas as pd
import time

start = time.time()
# importing modeling and data information from excel in dataframe
framegen=  pd.read_excel('./InputFiles/Example1.xlsx', 'framegen', header = 0, index_col=0)
seismic_defination= pd.read_excel('./InputFiles/Example1.xlsx', 'Seismic_Defination', header = 0, index_col=0)
load_combo= pd.read_excel('./InputFiles/Example1.xlsx', 'load_combinations', header = 0, index_col=0)

# Creating RC frame object for analysis
r1= RCF(nodes_details = None,member_details= None,boundarycondition= None,framegen= framegen, seismic_def= seismic_defination, load_combo= load_combo,  autoflooring= True) 

end1 = time.time()
#Pre processing the model
r1.preP()

end2 = time.time()
print ("Pre-Processing Time : ", (end2-end1))

"""If using VS Code or JupiterNotebook: models can be viewed by `model.show()` otherwise save model as html `model.write_html("./name.html")`
"""
# Initial model
model = r1.model3D()


# Deleting few members 
mem_list = [121,122,123,125,126,127,129,130,131,111,112,113,114,115,116,138,144,150,156,162,168,174,180,186,99,100,103,104,107,108,89,90,91,92,137,143,149,155,161,167,67,68,77,81,85,136,148,142]
r1.changeFrame(member= mem_list, delete= True)

#Changing width and depth of all members to 300 X 500 and udl of -10kN/m in y-direction on all members
r1.changeFrame(member= 'all',width= 300, depth= 500)
r1.changeFrame(member= 'beam', yudl= -10)

#Pre processing the model
r1.preP()

#Changing the floor thickness and loads 
# r1.changeFL(thickness= 100, LL= -5 , FF=-10, WP=0)

end3 = time.time()

# Performing Analysis
r1.RCanalysis()

end4 = time.time()
print ("Analysis Time : ", (end4-end3))

print ("Total Time : ", (end4-start))
# # Getting Reactions
reactions= r1.reactions()

# Getting nodal displacment
Ndisp= r1.Gdisp()

# Getting Shear Force and Bending Moment Diagram of Member with ID- 1
sfbmd= r1.sfbmd(47)
# sfbmd.write_html("./BMSF60.html")          

# Getting Material Properties
material_properties= r1.Mproperties()

# Getting design member forces
# dsg_memF = r1.maxmemF()



# seismicD= r1.seismicD()

# drift1= r1.Sdrift()

# Sshear_force =  r1.seismicS()

# defG= r1.defG(60) 
# beamsD= r1.beamsD()
# colD= r1.columnsD()
# nodesD= r1.nodesD()

# with pd.ExcelWriter('output_EXAMPLE1.xlsx') as writer:  
#     reactions.to_excel(writer, sheet_name='reactions')
#     Ndisp.to_excel(writer, sheet_name='nodal displacement')
#     material_properties.to_excel(writer, sheet_name='properties')
#     seismicD.to_excel(writer, sheet_name='seismic detail')
#     drift1.to_excel(writer, sheet_name='drift')
#     Sshear_force.to_excel(writer, sheet_name='Seismic Shear')
#     beamsD.to_excel(writer, sheet_name='Beams detail')
#     colD.to_excel(writer, sheet_name='columns detail')
#     nodesD.to_excel(writer, sheet_name='Nodes detail')