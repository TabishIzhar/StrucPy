# Example 4: Analysis of 5-story irregular building with floor loads and self weight using :class:`StrucPy.RCFA.RCF`. Floor Loads generated using :class:`StrucPy.RCFA.RCF.autoflooring`, and changes are made using inbuilt method :class:`StrucPy.RCFA.RCF.changeFL`. Load Combination (1.5 DL + 1.5 LL) is used.

from StrucPy.RCFA import RCF
import pandas as pd
import plotly


# Importing Input Data from Excel File (Note: Change the Path as per the location of File)
member_details= pd.read_excel('./InputFiles/Testfile4.xlsx', 'members', header = 0, index_col=0)
nodes_details= pd.read_excel('./InputFiles/Testfile4.xlsx', 'nodes', header = 0, index_col=0)
boundcond = pd.read_excel('./InputFiles/Testfile4.xlsx', 'boundary', header = 0, index_col=0)
load_combo= pd.read_excel('./InputFiles/Testfile4.xlsx', 'load_combinations', header = 0, index_col=0)

# Check "boundcond" for defining different boundary condition

#Creating RCF Object for structure
r1= RCF(nodes_details,member_details,boundcond, load_combo= load_combo, autoflooring= True) 

#Pre processing the model
r1.preP()

# View Floor Details (includes floor thickness, Floor Finish Loads, Live Loads and Water proofing Loads in kN/m2)
floor_load_details= r1.floorD()


# Changing all floor loads with  Live Load as 50kN/m2 and Floor thickness of 1000mm
r1.changeFL(thickness= 1000, LL=-50, FF=0)

# Performing Analysis
r1.RCanalysis()

# View 3D Model of Structure
view_3Dmodel= r1.model3D()

view_3Dmodel.show()                   # If using VS Code or JupiterNotebook
view_3Dmodel.write_html("./plots/Ex4_model3D.html")              # To save model as html in your current directory


# Getting Reactions
base_reactions= r1.reactions()


# Getting Nodal Displacements
nodal_displacements= r1.Gdisp()


# View 3D Deflected Shape of Structure
view_structure_deflected_shape= r1.def3D()

view_structure_deflected_shape.show()                              # If using VS Code or JupiterNotebook
view_structure_deflected_shape.write_html("./plots/Ex4_defShape_model3D.html")              # To save model as html in your current directory


# Getting Shear Force and Bending Moment Diagram of Member with ID- 756
sfbmd= r1.sfbmd(12)

sfbmd.show()                                     # If using VS Code or JupiterNotebook
sfbmd.write_html("./plots/Ex4_SFBMD.html")              # To save model as html in your current directory

# Getting Deflected Shape in Local Coordinate System of Member with ID- 756
defL= r1.defL(12)

defL.show()                                     # If using VS Code or JupiterNotebook
defL.write_html("./plots/Ex4_defL.html")              # To save model as html in your current directory


# Getting Deflected Shape in Global Coordinate System of Member with ID- 756
defG= r1.defG(12)

defG.show()                                     # If using VS Code or JupiterNotebook
defG.write_html("./plots/Ex4_defG.html")             # To save model as html in your current directory

# Getting Material Properties
material_properties= r1.Mproperties()

# Getting Details of All Beam Members
beams_details= r1.beamsD()

# Getting Details of All Column Members
columns_details= r1.columnsD()

# Getting Details of All Nodes (Joints)
nodes_deatils= r1.nodesD()

# Getting Data Details of All Member Forces in Every Member
member_forces_SF_BM_in_all_direction= r1.memF()

# Getting Data Details of Maximum Forces in Every Member
max_member_forces_in_all_direction= r1.MaxmemF()

# Getting Data Details of Deflection of Every Member in Local Coordinate System
deflection_local_coordinate_data= r1.defLD()

# Getting Data Details of Deflection of Every Member in Global Coordinate System
deflection_global_coordinate_data= r1.defGD()

# Getting Global Stiffness Matrix of Structure in Global Coordinate System
get_Global_stiffness_of_structure= r1.GlobalK()    