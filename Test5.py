#Example 5: Static seismic analysis of a 5-story irregular building with floor loads and self weight using :class:`StrucPy.RCFA.RCF`. Floor Loads generated using method autoflooring, and changes are made using inbuilt method. Seismic force is applied in x-direction. Load Combination is used: 1.5 DL + 1.2 EQx.


from StrucPy.RCFA import RCF
import pandas as pd
import plotly

# Importing Input Data from Excel File (Note: Change the Path as per the location of File)
member_details= pd.read_excel('./InputFiles/Example5.xlsx', 'members', header = 0, index_col=0)
nodes_details= pd.read_excel('./InputFiles/Example5.xlsx', 'nodes', header = 0, index_col=0)
boundcond = pd.read_excel('./InputFiles/Example5.xlsx', 'boundary', header = 0, index_col=0)
load_combo= pd.read_excel('./InputFiles/Example5.xlsx', 'load_combinations', header = 0, index_col=0)
seismic_defination= pd.read_excel('./InputFiles/Example5.xlsx', 'Seismic_Defination', header = 0, index_col=0)

# Check "seismic_defination" for defining seismic defination

# Creating RCF object for structure analysis
r1= RCF(nodes_details,member_details,boundcond, load_combo= load_combo, autoflooring= True, seismic_def= seismic_defination)

#Pre processing the model
r1.preP()

# View Floor Details (includes floor thickness, Floor Finish Loads, Live Loads and Water proofing Loads in kN/m2)
floor_load_details= r1.floorD()


# Changing all floor loads with Floor Finish load as 50kN/m2
r1.changeFL(thickness= 0, LL=0, FF=-50)

# Performing Analysis
r1.RCanalysis()

# View 3D Model of Structure
view_3Dmodel= r1.model3D()
view_3Dmodel.write_html("./plots/Ex5_model3D.html")              # To save model as html in your current directory

# Getting Reactions
base_reactions= r1.reactions()


# Getting Nodal Displacements
nodal_displacements= r1.Gdisp()

#Getting details of seismic calculation.
seismicD= r1.seismicD()

#Getting details of seismic shear.
seismic_shear= r1.seismicS()

#Getting details of stroy drift.
drift= r1.Sdrift()    

# To generate 3D Deflection Animation of Structure 

view_deflection_animation= r1.aniDef()
view_deflection_animation.write_html("./plots/Ex5_animation.html")              # To save model as html in your current directory

# View 3D Deflected Shape of Structure
view_structure_deflected_shape= r1.def3D()

view_structure_deflected_shape.show()                              # If using VS Code or JupiterNotebook
view_structure_deflected_shape.write_html("./plots/Ex5_defShape_model3D.html")              # To save model as html in your current directory


# Getting Shear Force and Bending Moment Diagram of Member with ID- 756
sfbmd= r1.sfbmd(12)

sfbmd.show()                                     # If using VS Code or JupiterNotebook
sfbmd.write_html("./plots/Ex5_SFBMD.html")              # To save model as html in your current directory

# Getting Deflected Shape in Local Coordinate System of Member with ID- 756
defL= r1.defL(12)

defL.show()                                     # If using VS Code or JupiterNotebook
defL.write_html("./plots/Ex5_defL.html")              # To save model as html in your current directory


# Getting Deflected Shape in Global Coordinate System of Member with ID- 756
defG= r1.defG(12)

defG.show()                                     # If using VS Code or JupiterNotebook
defG.write_html("./plots/Ex5_defG.html")             # To save model as html in your current directory