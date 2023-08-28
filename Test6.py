#Example 6: Changing the material properties of members of "RCFA Examples:Example 5".


from StrucPy.RCFA import RCF
import pandas as pd

# Importing Input Data from Excel File (Note: Change the Path as per the location of File)
member_details= pd.read_excel('./InputFiles/Testfile6.xlsx', 'members', header = 0, index_col=0)
nodes_details= pd.read_excel('./InputFiles/Testfile6.xlsx', 'nodes', header = 0, index_col=0)
boundcond = pd.read_excel('./InputFiles/Testfile6.xlsx', 'boundary', header = 0, index_col=0)
load_combo= pd.read_excel('./InputFiles/Testfile6.xlsx', 'load_combinations', header = 0, index_col=0)
seismic_defination= pd.read_excel('./InputFiles/Testfile6.xlsx', 'Seismic_Defination', header = 0, index_col=0)

grade_concrete= 30              # If want same calculation but just different concrete grade, grade_concrete can be used.

material_properties= pd.read_excel('./InputFiles/Testfile6.xlsx', 'Mproperties', header = 0, index_col=0)


# Check "seismic_defination" for defining seismic defination

# Creating RCF object "r1" for structure analysis just changing the grade of concrete, entire calculation for estimation of young modulus, modulus of rigidity etc. remains same 
r1= RCF(nodes_details,member_details,boundcond, load_combo= load_combo, autoflooring= True, seismic_def= seismic_defination, grade_conc= grade_concrete)


# Creating RCF object "r2" for structure analysis, completely changing the material properties
r2= RCF(nodes_details,member_details,boundcond, load_combo= load_combo, autoflooring= True, seismic_def= seismic_defination, properties= material_properties )


#Pre processing the model
r1.preP()

r2.preP()

# View Floor Details (includes floor thickness, Floor Finish Loads, Live Loads and Water proofing Loads in kN/m2)
floor_load_details= r1.floorD()


# Changing all floor loads with Floor Finish load as 50kN/m2
r1.changeFL(thickness= 0, LL=0, FF=-50)

r2.changeFL(thickness= 0, LL=0, FF=-50)


# Performing Analysis
r1.RCanalysis()
r2.RCanalysis()


# Getting Reactions
base_reactions_r1= r1.reactions()
base_reactions_r2= r2.reactions()

#Getting Material Properties
Mproperties_r1= r1.Mproperties()

Mproperties_r2= r2.Mproperties()
