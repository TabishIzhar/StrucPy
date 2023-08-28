#Example 1: Analysis of fixed Beam With multiple point loads using :class:`StrucPy.RCFA.RCF`.

from StrucPy.RCFA import RCF
import pandas as pd
import plotly

framegen=  pd.read_excel('./InputFiles/Example1.xlsx', 'framegen', header = 0, index_col=0)
seismic_defination= pd.read_excel('./InputFiles/Example1.xlsx', 'Seismic_Defination', header = 0, index_col=0)
load_combo= pd.read_excel('./InputFiles/Example1.xlsx', 'load_combinations', header = 0, index_col=0)

# Creating RC frame object for analysis
r1= RCF(nodes_details = None,member_details= None,boundrycondition= None,framegen= framegen, seismic_def= seismic_defination, load_combo= load_combo,  autoflooring= True )

#Pre processing the model
r1.preP()

# Performing Analysis
r1.RCanalysis()

# Getting Reactions
reactions= r1.reactions()

# Getting Shear Force and Bending Moment Diagram of Member with ID- 1
sfbmd= r1.sfbmd(1)
sfbmd.show()                                     # If using VS Code or JupiterNotebook
sfbmd.write_html("./plots/Ex1_SFBMD.html")              # To save model as html in your current directory


# Getting Deflected Shape in Local Coordinate System of Member with ID- 1
defL= r1.defL(1)
defL.show()                                     # If using VS Code or JupiterNotebook
defL.write_html("./plots/Ex1_defL.html")              # To save model as html in your current directory


# Getting Deflected Shape in Global Coordinate System of Member with ID- 1
defG= r1.defG(1)
defG.show()                                     # If using VS Code or JupiterNotebook
defG.write_html("./plots/Ex1_defG.html")             # To save model as html in your current directory

# Getting Material Properties
material_properties= r1.Mproperties()