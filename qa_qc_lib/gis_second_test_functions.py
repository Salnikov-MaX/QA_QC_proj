import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import re
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def poro_test(data_rigis, data_kern, rigis_name, kern_name):
    data_rigis = data_rigis[['MD', rigis_name]]
    data_kern = data_kern[['MD', kern_name]]
    merged, k, b, r = preparing(data_rigis, data_kern, rigis_name, kern_name)
    properties_vizualization(merged[rigis_name], merged[kern_name], name='poro', k=k, b=b, r=r)

def perm_test(data_rigis, data_kern, rigis_name, kern_name):
    data_rigis = data_rigis[['MD', rigis_name]]
    data_kern = data_kern[['MD', kern_name]]
    merged, k, b, r = preparing(data_rigis, data_kern, rigis_name, kern_name)
    properties_vizualization(merged[rigis_name], merged[kern_name], name='perm', k=k, b=b, r=r)

def poroeff_test(data_rigis, data_kern, rigis_name, kern_name):
    data_rigis = data_rigis[['MD', rigis_name]]
    data_kern = data_kern[['MD', kern_name]]
    merged, k, b, r = preparing(data_rigis, data_kern, rigis_name, kern_name)
    properties_vizualization(merged[rigis_name], merged[kern_name], name='poroeff', k=k, b=b, r=r)

def saturation_test(data_rigis, data_kern, rigis_name, kern_name):
    """Данных по насыщенности нет, но принцип такой же"""
    data_rigis = data_rigis[['MD', rigis_name]]
    data_kern = data_kern[['MD', kern_name]]
    merged, k, b, r = preparing(data_rigis, data_kern, rigis_name, kern_name)
    properties_vizualization(merged[rigis_name], merged[kern_name], name='saturation', k=k, b=b, r=r)

def lithology_test(data_rigis, data_kern, rigis_name, kern_name):
    data_rigis = data_rigis[['MD', rigis_name]]
    data_kern = data_kern[['MD', kern_name]]
    data = pd.merge_asof(left = data_kern, right = data_rigis, on = 'MD', direction='nearest')
    data = data.dropna()
    lithology_vizualization(data['MD'], data[kern_name], data[rigis_name])

def preparing(data_rigis, data_kern, rigis_name, kern_name):
    merged = data_kern.merge(data_rigis, on=['MD', 'MD'], how='outer', sort=True)
    merged[rigis_name].interpolate(inplace=True)
    merged = merged.dropna()
    k, b, r, p, se = stats.linregress(merged[kern_name], merged[rigis_name])
    return merged, k, b, r

def linear_func(x, a, b):
        return a * x + b

def properties_vizualization(data_kern, data_rigis, name, k, b, r):
    y_new = linear_func(data_rigis, k, b)
    plt.scatter(data_rigis, data_kern)
    plt.plot(sorted(data_rigis), sorted(y_new), c='orange')
    maximum = max(max(data_rigis), max(data_kern))
    plt.plot([0, maximum], [0, maximum], '--', c='g')
    plt.xlim(0.0, maximum*1.1)
    plt.ylim(0.0, maximum*1.1)
    plt.text(0.05, 0.05, 'y=' + str(round(k, 3)) + '*rigis+' + str(round(b,3))  + ', коэффициент R^2 = ' + str(round(r**2, 3)))
    if name == 'poro':
        plt.xlabel('РИГИС пористости, д.е.')
        plt.ylabel('Пористость по керну, д.е.')
        plt.title('Пористость')
    elif name == 'poroeff':
        plt.xlabel('РИГИС эффективной пористости, д.е.')
        plt.ylabel('Эффективная пористость по керну, д.е.')
        plt.title('Эффективная пористость')
    elif name == 'perm':
        plt.xlabel('РИГИС проницаемости, мД')
        plt.ylabel('Проницаемость по керну, мД')
        plt.title('Проницаемость')
        plt.xlim(0.01, maximum*2)
        plt.ylim(0.01, maximum*2)
        plt.xscale('log')
        plt.yscale('log')
    elif name == 'saturation':
        plt.xlabel('РИГИС водонасыщенности, %')
        plt.ylabel('Водонасыщенность по керну, %')
        plt.title('Водонасыщенность')
    plt.show()

def lithology_vizualization(depth, kern_lithology, litho):
        column_width = [0.5, 0.5]
        column_titles = ['Керн', 'ГИС']
        colors = [
            "rgb(255, 250, 205)",
            "rgb(139, 69, 19)",
            "rgb(0, 0, 0)",
            "rgb(255, 250, 205)",
            "rgb(240, 230, 140)",
            ] 
        colorscale=[
                (0.0, colors[3]),
                (0.25, colors[0]),
                (0.5, colors[4]),
                (0.75, colors[1]),
                (1, colors[2]),
            ]
        kern_lithology = [(x-0)/(4-0) for x in kern_lithology]
        litho = [(x-0)/(4-0) for x in litho]
        
        fig = make_subplots(rows=1, cols=2, column_widths=column_width, subplot_titles=column_titles, horizontal_spacing=0.1)
        fig.add_trace(go.Heatmap(y=depth, z=np.array(kern_lithology).reshape(-1, 1), hovertext=np.array(kern_lithology).reshape(-1, 1), colorscale=colorscale, showscale=False, zmin=0, zmax=1, hovertemplate='Глубина: %{y} <br>Литология: %{hovertext}'), row=1, col=1)
        fig.add_trace(go.Heatmap(y=depth, z=np.array(litho).reshape(-1, 1), hovertext=np.array(litho).reshape(-1, 1), colorscale=colorscale, showscale=False, zmin=0, zmax=1, hovertemplate='Глубина: %{y} <br>Литология: %{hovertext}'), row=1, col=2)
        fig.update_yaxes(autorange="reversed", row=1, col=1, matches='y')
        fig.update_yaxes(autorange="reversed", row=1, col=2, matches='y')

        fig.update_layout(height=1000, width=600, yaxis_range=[depth[0], list(depth)[-1]]) 
        fig.show()
    
    