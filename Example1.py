#Example 1: Analysis of fixed Beam With multiple point loads using :class:`StrucPy.RCFA.RCF`.

from StrucPy.RCFA import RCF
import pandas as pd
import plotly

framegen=  pd.read_excel('./InputFiles/Example1.xlsx', 'framegen', header = 0, index_col=0)
seismic_defination= pd.read_excel('./InputFiles/Example1.xlsx', 'Seismic_Defination', header = 0, index_col=0)
load_combo= pd.read_excel('./InputFiles/Example1.xlsx', 'load_combinations', header = 0, index_col=0)

# Creating RC frame object for analysis
r1= RCF(nodes_details = None,member_details= None,boundarycondition= None,framegen= framegen, seismic_def= seismic_defination, load_combo= load_combo,  autoflooring= True)


"""If using VS Code or JupiterNotebook: models can be viewed by `model.show()` otherwise save model as html `model.write_html("./name.html")`
"""

#Veiw the model in order to edit members or floors


#Pre processing the model
r1.preP()

model = r1.model3D()
model.show()

# Deleting few members 

# Performing Analysis
r1.RCanalysis()

# Getting Reactions
reactions= r1.reactions()

# Getting Shear Force and Bending Moment Diagram of Member with ID- 1
sfbmd= r1.sfbmd(1)
# sfbmd.show()             # If using VS Code or JupiterNotebook


# Getting Material Properties
material_properties= r1.Mproperties()