import plotly.graph_objects as go
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import math
from os import path



def format_int_K(n: int):
    return str(n)[::-1].replace("000", "K")[::-1]


def visualize_results(res, drift_hist, dataset_name, replacement_hist, metrics=None, out_dir=None):
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

    stream_len = len(x)
    x = x[0::10]
    res = res[0::10]
    num_parts = get_num_parts(stream_len)

    # Create traces
    fig = make_subplots(rows=num_parts, cols=1,
        x_title='Consecutive samples in the data stream',
        y_title='Metrics values'
    )

    x_parts, res_parts = get_partitions(num_parts, x, res)

    showlegend = True
    shown_drifts = set()
    for p in range(num_parts):
        x_part = x_parts[p]
        res_part = res_parts[p]
        dmax = res_part.values.max() 
        dmin = res_part.values.min()

        dmax = dmax + 0.1 * dmax
        dmin = dmin - 0.1 * dmin

        for m in res.columns:
            l = 0
            if 'f1' in m: l += 2
            if 'acc' in m: l += 4
            if 'preq' in m: l += 1
            c = colors_m[l]
            fig.add_trace(go.Scatter(x=x_part, y=res_part[m], showlegend=showlegend,
                                legendgroup="group",
                                legendgrouptitle_text="Metrics",
                                line=dict(color=c, width=2), name=get_metric_name(m)), row=p+1, col=1)
        if p == 0:
            fig.update_layout(
                xaxis = dict(
                    tickmode = 'linear',
                    tick0 = 0,
                    dtick = 10_000,
                )
            )
        else:
            fig.update_layout({
                f'xaxis{p + 1}': dict(
                    tickmode = 'linear',
                    tick0 = x_part[0],
                    dtick = 10_000,
                )
            })

        for i in range(num_drift_types):
            drift_type = list(drift_hist.keys())[i]
            indices = drift_hist[drift_type]
            color = colors[i]
            indices = [int(d) for d in indices if int(d) >= x_part[0] and int(d) < x_part[-1]]
            for idx in indices:
                showlegend_drift = False if drift_type in shown_drifts else True
                shown_drifts.add(drift_type)
                fig.add_trace(go.Scatter(x=[idx, idx], 
                                y=[dmin, dmax], 
                                mode='lines', 
                                line=dict(color=color, width=2, dash='dash'),
                                legendgroup="group2",
                                legendgrouptitle_text="Drift type",
                                showlegend=showlegend_drift, 
                                name=f"{drift_type.split('_')[0]}"
                                # text=[format_int_K(idx), ""],
                                # textposition=[f"bottom {pos}", "top center"],
                                # textfont={
                                #         "color": color,
                                #         "size": 8
                                #     },
                                ), row=p+1, col=1
                            )
        showlegend = False

        indices = [int(d) for d in replacement_hist if int(d) >= x_part[0] and int(d) < x_part[-1]]
        for idx in indices:
            fig.add_trace(go.Scatter(x=[idx, idx], 
                            y=[dmin, dmax - (dmax - dmin)*0.15], 
                            mode='lines+text', 
                            line=dict(color="red", width=0, dash='dot'),
                            showlegend=False, 
                            legendgroup="group2",
                            legendgrouptitle_text="Drift type",
                            text=["", "R"],
                            textposition=["bottom center", "top center"],
                            textfont={
                                        "color": "red",
                                        "size": 14
                            },
                            hoverinfo=None,
                            hovertext=f"Replacement index: {idx}"
                            ), row=p+1, col=1
                        )
        
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    #fig.update_layout({"title": f"Results for {dataset_name} dataset"})
    
    fig.update_layout(
        autosize=False,
        width=800,
        height=800,
    )
    if out_dir is not None:
        fig.write_image(file=path.join(out_dir, 'fig.pdf'), format='pdf')

    fig.show()
    return fig



def get_partitions(num_partitions: int, x, res):
    k, m = divmod(len(x), num_partitions)
    x_parts = list((x[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_partitions)))
    res_parts = list((res[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_partitions)))
    return x_parts, res_parts


def get_num_parts(n: int):

    k = math.ceil(n / get_interval_length())
    return k

def get_interval_length():
    return 100_000


def get_metric_name(metric_name: str):
    dt = metric_name.lower()
    name = ""
    if "f1" in dt and "macro" in dt:
        name = "F1 macro"
    if "acc" in dt:
        name = "Accuracy"
    if "kappa" in dt:
        name = "Cohen-Kappa"
    if "preq" in dt:
        window_size = dt.split("preq_")[-1]
        name += f" (preq. {window_size})"
    return name
