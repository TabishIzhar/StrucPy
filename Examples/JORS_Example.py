from StrucPy.RCFA import RCF
import pandas as pd

# importing modeling and data information from excel in dataframe
framegen=  pd.read_excel('../InputFiles/JORS_Example.xlsx', 'framegen', header = 0, index_col=0)
seismic_defination= pd.read_excel('../InputFiles/JORS_Example.xlsx', 'Seismic_Defination', header = 0, index_col=0)
load_combo= pd.read_excel('../InputFiles/JORS_Example.xlsx', 'load_combinations', header = 0, index_col=0)

# Creating RC frame object for analysis
r1= RCF(nodes_details = None,member_details= None,boundarycondition= None,framegen= framegen,seismic_def=seismic_defination, load_combo= load_combo,  autoflooring= True) 

#Pre processing the model
r1.preP()

"""If using VS Code or JupiterNotebook: models can be viewed by `model.show()` otherwise save model as html `model.write_html("./name.html")`
"""

# Initial model
model = r1.model3D()

# Deleting few members 
mem_list = [121,122,123,125,126,127,129,130,131,111,112,113,114,115,116,138,144,150,156,162,168,174,180,186,99,100,103,104,107,108,89,90,91,92,137,143,149,155,161,167,67,68,77,81,85,136,148,142]
r1.changeFrame(member= mem_list, delete= True)

#Changing width and depth of all members to 400 X 400 and udl of -10kN/m in y-direction on all members
r1.changeFrame(member= 'all',width= 400, depth= 400)
r1.changeFrame(member= 'beam', yudl= -10)

#Pre processing the model
r1.preP()

#Changing the floor thickness and loads 
r1.changeFL(thickness= 100, LL= -3 , FF=-5, WP=0)

# Performing Analysis
r1.RCanalysis()

# Getting Reactions
reactions= r1.reactions()

# Getting nodal displacment
Ndisp= r1.Gdisp()

# Getting Shear Force and Bending Moment Diagram of Member with ID- 1
sfbmd= r1.sfbmd(47)
# sfbmd.write_html("./BMSF47.html")          

# Getting Material Properties
material_properties= r1.Mproperties()

# Getting seismic caluclation
seismicD= r1.seismicD()

# Getting seismic shear
seismicS= r1.seismicS()

# Getting story average displacement and drift
Sdrift= r1.Sdrift()

# Getting deflection of Member 60 in global coordinate system
defG= r1.defG(47) 

# Getting beams detail of RC frame
beamsD= r1.beamsD()

# Getting columns detail of RC frame
colD= r1.columnsD()

# Getting nodes detail of RC frame
nodesD= r1.nodesD()

# Getting Shear Force and Bending Moment Data of RC frame
design_forces= r1.maxmemF()

# Getting deflection data in local coordinate system of RC frame
defLD= r1.defLD()

# Getting deflection data in global coordinate system of RC frame
defGD= r1.defGD()

# Getting floor details of RC frame
floord= r1.floorD()

# Getting Shear Force and Bending Moment Data of RC frame
memF= r1.memF()

# Getting maximum member forces of all RC frame members
maxmemF= r1.maxmemF()

# Getting maximum member deflection of all RC frame members in local coordinate system
maxdefL= r1.maxdefL()


with pd.ExcelWriter('output_EXAMPLE1.xlsx') as writer:  
    reactions.to_excel(writer, sheet_name='reactions')
    Ndisp.to_excel(writer, sheet_name='nodal displacement')
    material_properties.to_excel(writer, sheet_name='properties')
    floord.to_excel(writer, sheet_name='floor detail')    
    seismicD.to_excel(writer, sheet_name='seismic detail')
    seismicS.to_excel(writer, sheet_name='seismic shear')
    Sdrift.to_excel(writer, sheet_name='story drift')
    beamsD.to_excel(writer, sheet_name='beams detail')
    colD.to_excel(writer, sheet_name='columns detail')
    nodesD.to_excel(writer, sheet_name='nodes detail')
    maxmemF.to_excel(writer, sheet_name='maximum member forces')
   



