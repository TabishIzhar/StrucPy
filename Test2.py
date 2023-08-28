
#Example 2: Analysis of simple supported beam with uniformly distributed loads (UDL) using :class:`StrucPy.RCFA.RCF`.


from StrucPy.RCFA import RCF
import pandas as pd
import plotly


# Importing Input Data from Excel File (Note: Change the Path as per the location of File)
member_details= pd.read_excel('./InputFiles/Testfile2.xlsx', 'members', header = 0, index_col=0)
nodes_details= pd.read_excel('./InputFiles/Testfile2.xlsx', 'nodes', header = 0, index_col=0)
boundcond = pd.read_excel('./InputFiles/Testfile2.xlsx', 'boundary', header = 0, index_col=0)

# Check "boundcond" for defining different boundary condition

# Self weight is being ignored
r1= RCF(nodes_details,member_details,boundcond, self_weight=False)

#Pre processing the model
r1.preP()

# Performing Analysis
r1.RCanalysis()

# Getting Reactions
reactions= r1.reactions()

# Getting Shear Force and Bending Moment Diagram of Member with ID- 1
sfbmd= r1.sfbmd(1)
sfbmd.show()                                     # If using VS Code or JupiterNotebook
sfbmd.write_html("./plots/Ex2_SFBMD.html")              # To save model as html in your current directory


# Getting Deflected Shape in Local Coordinate System of Member with ID- 1
defL= r1.defL(1)
defL.show()                                     # If using VS Code or JupiterNotebook
defL.write_html("./plots/Ex2_defL.html")              # To save model as html in your current directory


# Getting Deflected Shape in Global Coordinate System of Member with ID- 1
defG= r1.defG(1)
defG.show()                                     # If using VS Code or JupiterNotebook
defG.write_html("./plots/Ex2_defG.html")             # To save model as html in your current directory

# Getting Material Properties
material_properties= r1.Mproperties()