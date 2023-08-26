from StrucPy.RCFA import RCF
import pandas as pd
import plotly


framegen=  pd.read_excel('./Example1.xlsx', 'framegen', header = 0, index_col=0)
seismic_defination= pd.read_excel('./Example1.xlsx', 'Seismic_Defination', header = 0, index_col=0)
load_combo= pd.read_excel('./Example1.xlsx', 'load_combinations', header = 0, index_col=0)


r1= RCF(nodes_details = None,member_details= None,boundrycondition= None,framegen= framegen, seismic_def= seismic_defination, load_combo= load_combo,  autoflooring= True )

r1.preP()

f1,f2,f3,fig,am,af= r1.model3D()

fig.write_html("./ExMOdel.html")