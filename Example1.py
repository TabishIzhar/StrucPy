#Example 1: Analysis of fixed Beam With multiple point loads using :class:`StrucPy.RCFA.RCF`.

from StrucPy.RCFA import RCF
import pandas as pd
import plotly

# Importing Input Data from Excel File (Note: Change the Path os the file as per the location of "file" and "working directory")
member_details= pd.read_excel('InputFiles/Testfile1.xlsx', 'members', header = 0, index_col=0)
nodes_details= pd.read_excel('InputFiles/Testfile1.xlsx', 'nodes', header = 0, index_col=0)
boundcond = pd.read_excel('InputFiles/Testfile1.xlsx', 'boundary', header = 0, index_col=0)
point_loads= pd.read_excel('InputFiles/Testfile1.xlsx', 'point_loads', header = 0, index_col=0)

# Self weight is being ignored
r1= RCF(nodes_details,member_details,boundcond, point_loads= point_loads, self_weight=False)

#Pre processing the model
r1.preP()

# Performing Analysis
r1.RCanalysis()

# Getting Reactions
reactions= r1.reactions()

# Getting Shear Force and Bending Moment Diagram of Member with ID- 1
sfbmd= r1.sfbmd(1)
sfbmd.show()                         # If using VS Code or JupiterNotebook
sfbmd.write_html("./plots/Ex1_SFBMD.html")    # To save model as html in your current directory


# Getting Deflected Shape in Local Coordinate System of Member with ID- 1
defL= r1.defL(1)
defL.show()                      # If using VS Code or JupiterNotebook
defL.write_html("./plots/Ex1_defL.html")     # To save model as html in your current directory


# Getting Deflected Shape in Global Coordinate System of Member with ID- 1
defG= r1.defG(1)
defG.show()                     # If using VS Code or JupiterNotebook
defG.write_html("./plots/Ex1_defG.html")     # To save model as html in your current directory

# Getting Material Properties
material_properties= r1.Mproperties()