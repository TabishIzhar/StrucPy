import pandas as pd
import plotly.express as px
import textwrap

df1=  pd.read_excel('./DataBoxPlot.xlsx', 'REACTPLOT', header = 0)

df2=  pd.read_excel('./DataBoxPlot.xlsx', 'DISPPLOT', header = 0)


fig1 = px.box(df1, x= "Reactions", y="Difference in Values", color="Load Combination",
             notched=True, # used notched shape
             points = False

            )

fig1.update_xaxes(tickfont=dict(family='Arial', color='black', size=22))
fig1.update_yaxes(tickfont=dict(family='Arial', color='black', size=22))

fig1.update_xaxes(showgrid=True, gridwidth=0.1, gridcolor='white', title_standoff = 40)
fig1.update_yaxes(showgrid=True, gridwidth=0.1, gridcolor='white', title_standoff = 40) #LightPink

fig1.update_yaxes(tick0= 0, dtick=0.5)

fig1.update_xaxes(ticks="outside", tickwidth=1, tickcolor='black', ticklen=10) #ticks="inside",
fig1.update_yaxes(ticks="outside",tickwidth=1, tickcolor='black', ticklen=10)

fig1.update_yaxes(ticklabelstep=2)

fig1.update_xaxes(title_font_family="Arial", title_font_size= 30)
fig1.update_yaxes(title_font_family="Arial", title_font_size= 30)

fig1.update_layout(
    legend_title_text = "Load Combinations",
    legend_font=dict(
        family="Arial, monospace",
        size=18,
            ),
)

fig1.update_layout(height=1200, width=1800)






fig2 = px.box(df2, x= "Displacements", y="Difference in Values", color="Load Combination",
             notched=True, # used notched shape
             points = False
            )


fig2.update_xaxes(tickfont=dict(family='Arial', color='black', size=22))
fig2.update_yaxes(tickfont=dict(family='Arial', color='black', size=22))

fig2.update_xaxes(showgrid=True, gridwidth=0.1, gridcolor='white', title_standoff = 40)
fig2.update_yaxes(showgrid=True, gridwidth=0.1, gridcolor='white', title_standoff = 40) #LightPink


fig2.update_yaxes(tick0= 0, dtick=0.5)

fig2.update_xaxes(ticks="outside", tickwidth=1, tickcolor='black', ticklen=10) #ticks="inside",
fig2.update_yaxes(ticks="outside",tickwidth=1, tickcolor='black', ticklen=10)

fig2.update_yaxes(ticklabelstep=2)

fig2.update_xaxes(title_font_family="Arial", title_font_size= 30)
fig2.update_yaxes(title_font_family="Arial", title_font_size= 30)

fig2.update_layout(
    legend_title_text = "Load Combinations",
    legend_font=dict(
        family="Arial, monospace",
        size=18,
            ),
)

fig2.update_layout(height=1200, width=1800)







fig1.write_html("./boxplotReactions.html")
fig1.show()

fig2.write_html("./boxplotDisplacements.html")
fig2.show()