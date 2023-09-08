import pandas as pd
import plotly.express as px

df1=  pd.read_excel('./DataBoxPlot.xlsx', 'try2', header = 0)

df = px.data.tips()
fig = px.box(df, x="time", y="total_bill", color="smoker",
             notched=True, # used notched shape
             title="Box plot of total bill",
             hover_data=["day"] # add day column to hover data
            )
# fig.show()


fig1 = px.box(df1, x= "Reactions", y="Difference in Values", color="Load Combination",
             notched=True, # used notched shape
             # add day column to hover data
            )
fig1.update_layout(height=1000, width=1600)

fig1.update_yaxes(dtick=0.5)

fig1.update_layout(

    legend_title="Load Combinations",
    font=dict(
        family="Courier New, monospace",
        size=20,
        # color="RebeccaPurple"
    )
)
fig1.show()