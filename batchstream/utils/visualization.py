import plotly.graph_objects as go
import numpy as np
import plotly.express as px



def format_int_K(n: int):
    return str(n)[::-1].replace("000", "K")[::-1]


def visualize_results(res, drift_hist, dataset_name, metrics=None):
    x = np.array(res.index)
    if metrics == None:
        metrics = res.columns

    num_drift_types = len(drift_hist.keys())
    colors = px.colors.qualitative.D3
    while num_drift_types > len(colors):
        colors *= 2

    colors_m = px.colors.qualitative.G10

    for m in metrics:
        res = res[[c for c in res.columns if m in c]]

    x = x[0::10]
    res = res[0::10]

    dmax = res.values.max() 
    dmin = res.values.min()

    dmax = dmax + 0.1 * dmax
    dmin = dmin - 0.1 * dmin

    # Create traces
    fig = go.Figure()

    for m in res.columns:
        l = 0
        if 'f1' in m: l += 2
        if 'acc' in m: l += 4
        if 'preq' in m: l += 1
        c = colors_m[l]
        fig.add_trace(go.Scatter(x=x, y=res[m],
                            legendgroup="group",
                            legendgrouptitle_text="Metrics",
                            line=dict(color=c, width=2), name=m))

    for i in range(num_drift_types):
        drift_type = list(drift_hist.keys())[i]
        indices = drift_hist[drift_type]
        color = colors[i]
        showlegend = True
        k = 0
        for idx in indices:
            if k % 3 == 0: pos = "right"
            elif k % 3 == 1: pos = "center"
            else: pos = "left"
            fig.add_trace(go.Scatter(x=[idx, idx], 
                            y=[dmin, dmax], 
                            mode='lines', 
                            line=dict(color=color, width=2, dash='dash'),
                            legendgroup="group2",
                            legendgrouptitle_text="Drift type",
                            showlegend=showlegend, 
                            name=f"{drift_type.split('_')[0]}"
                            # text=[format_int_K(idx), ""],
                            # textposition=[f"bottom {pos}", "top center"],
                            # textfont={
                            #         "color": color,
                            #         "size": 8
                            #     },
                            ))
            showlegend = False
            k += 1
            
    fig.update_layout({"title": f"Results for {dataset_name} dataset"})

    fig.update_layout(
        autosize=False,
        width=1500,
        height=500,
        margin=dict(
            l=20,
            r=20,
            b=50,
            t=50,
            pad=4
        ),
    )

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    fig.update_layout(
        xaxis = dict(
            tickmode = 'linear',
            tick0 = 0,
            dtick = 10_000
        )
    )

    fig.update_xaxes(
        title_text = "Consecutive samples in the data stream",
        title_standoff = 25)
    
    fig.update_yaxes(
        title_text = "Metrics values",
        title_standoff = 25)
    
    fig.show()
