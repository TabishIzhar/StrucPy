import plotly.graph_objects as go
import textwrap
import pandas as pd

def customwrap(s,width=25):
    return "<br>".join(textwrap.wrap(s,width=width))


df1 = pd.read_excel('packF.xlsx','nodes', header = 0, index_col=0)
df2 = pd.read_excel('packF.xlsx','elements', header = 0, index_col=0)
df3=  pd.read_excel('packF.xlsx','boundary', header = 0, index_col=0)
df4=  pd.read_excel('packF.xlsx','seismic', header = 0, index_col=0)

g1 = df1.labels.map(customwrap)
g2 = df1.parents.map(customwrap)

g3 = df2.labels.map(customwrap)
g4 = df2.parents.map(customwrap)

g5 = df3.labels.map(customwrap)
g6 = df3.parents.map(customwrap)

g7 = df4.labels.map(customwrap)
g8 = df4.parents.map(customwrap)

dfF1 = pd.DataFrame(
    dict(ids=df1.ids, labels=g1, parents=g2)
)

dfF2 = pd.DataFrame(
    dict(ids=df2.ids, labels=g3, parents=g4)
)

dfF3 = pd.DataFrame(
    dict(ids=df3.ids, labels=g5, parents=g6)
)

g7.loc[3]= "<br>".join(textwrap.wrap(g7.loc[3],width=15))


dfF4 = pd.DataFrame(
    dict(ids=df4.ids, labels=g7, parents=g8)
)

dfF1.loc[1,"parents"]= ""
dfF2.loc[1,"parents"]= ""
dfF3.loc[1,"parents"]= ""
dfF4.loc[1,"parents"]= ""

fig = go.Figure()

fig.add_trace(go.Sunburst(
    ids=dfF1.ids,
    labels=dfF1.labels,
    parents=dfF1.parents,
    domain=dict(row=0, column=0),
    marker=dict(colors=['#faf3e8', '#d8ebae', '#c6def5', '#e3aac9', '#f5f389'], line= dict(color= '#000000')),
    insidetextorientation='auto',
    textfont=dict(size=18, color='#000000', family='Arial')
))

fig.add_trace(go.Sunburst(
    ids=dfF2.ids,
    labels=dfF2.labels,
    parents=dfF2.parents,
    domain=dict(row=0, column=1),
    marker=dict(colors=['#faf3e8', '#d8ebae', '#c6def5', '#e3aac9', '#f5f389', '#ebbcf7', '#c3facd', '#faa5d5', '#b1d9f2'], line= dict(color= '#000000')),
    insidetextorientation='auto',
    textfont=dict(size=18, color='#000000', family='Arial')
))

fig.add_trace(go.Sunburst(
    ids=dfF3.ids,
    labels=dfF3.labels,
    parents=dfF3.parents,
    domain=dict(row= 1, column=0),
    marker=dict(colors=['#faf3e8'], line= dict(color= '#000000')),
    insidetextorientation='auto',
    textfont=dict(size=18, color='#000000', family='Arial')
))

fig.add_trace(go.Sunburst(
    ids=dfF4.ids,
    labels=dfF4.labels,
    parents=dfF4.parents,
    domain=dict(row= 1, column=1),
    marker=dict(colors=['#faf3e8', '#b1eaf2','#f5f389'], line= dict(color= '#000000')),
    insidetextorientation='auto',
    textfont=dict(size=18, color='#000000', family='Arial')
))

# fig.update_yaxes(title_text="<b> a. Features of Instance nodes_details </b>",  row=1, col=1)   #layout_yaxis_range=[-(abs(min(sfy))+5),(abs(max(sfy))+5)],
# fig.update_yaxes(title_text="<b> b. Features of Instance member_details</b>", row=1, col=2)
# fig.update_yaxes(title_text="<b> c. Features of Instance boundarycondition</b>", row=2, col=1)
# fig.update_yaxes(title_text="<b> d. Features of Instance seismic_def</b>", row=2, col=2)


fig.update_layout(
    grid= dict(columns=2, rows=2),
    margin = dict(t=0, l=0, r=0, b=0)
)

# fig.update_layout(uniformtext=dict(minsize=16, mode='hide'))

fig.update_layout(height=1000, width=1000)
fig.write_html("./figure1.html")
fig.show()



# import plotly.express as px
# import pandas as pd




# df = pd.read_excel("/Users/stuff/Desktop/test.xlsx")

# g1 = df.Group1.map(customwrap)
# g2 = df.Group2.map(customwrap)
# g3 = df.Group3.map(customwrap)
# g4 = df.Group4.map(customwrap)
# my_values = df.Values

# df = pd.DataFrame(
#     dict(group1=g1, group2=g2, group3=g3, group4=g4, my_values=my_values)
# )
# fig = px.sunburst(df, path=['group1', 'group2', 'group3', 'group4'], values='my_values')

# fig.show()