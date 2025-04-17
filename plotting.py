from typing import Literal
from pymc_marketing.mmm import MMM
import arviz as az
import plotly.graph_objs as go
import pandas as pd

def plot_predictive(model: MMM, actuals: pd.Series, label = Literal["prior_predictive", "posterior_predicitive"])->None:

    dataset = model._get_group_predictive_data(
            group=label, original_scale=True
        )
    mean = dataset['y'].mean(dim=["chain", "draw"])
    likelihood_hdi_95 = az.hdi(ary=dataset, hdi_prob=0.95)[
                'y'
            ]
    likelihood_hdi_50 = az.hdi(ary=dataset, hdi_prob=0.50)[
                'y'
            ]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dataset.date, 
            y=actuals,
            name= "actual",
            showlegend=True
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dataset.date, 
            y=mean,
            name= "mean",
            showlegend=True
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dataset.date, 
            y=likelihood_hdi_95[:,0],
            name= "Lower bound(95%)",
            showlegend=True
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dataset.date, 
            y=likelihood_hdi_95[:,1],
            name= "Upper bound (95%)",
            showlegend=True,
            fill="tonexty"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dataset.date, 
            y=likelihood_hdi_50[:,0],
            name= "Lower bound(50%)",
            showlegend=True
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dataset.date, 
            y=likelihood_hdi_50[:,1],
            name= "Upper bound (50%)",
            showlegend=True,
            fill="tonexty"
        )
    )
    fig.update_layout(
    title=label,
    xaxis_title="date",
    yaxis_title="y",
    font=dict(
        family="Arial",
        size=20,
        color="green"
    )
    )
    fig.show()