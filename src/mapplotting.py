from matplotlib import colors
import matplotlib.pyplot as plt
import osmnx as ox
import defines

cols = ['k',        # Black
        '#4CBB17',        # Green
        '#ff6600',   # Orange
        'r',        # Red
        '#882d17']  # Brown

def select_second_color_scale(val):
    if val < 0:
        return cols[0]
    elif val <= 10:
        return cols[1]
    elif val <= 20:
        return cols[2]
    elif val <= 30:
        return cols[3]
    else:
        return cols[4]
    

def select_color(val, original, second_color_scale):
    if second_color_scale:
        return select_second_color_scale(val)
    
    if val < 0:
        return cols[0] if original else cols[1]
    elif val <= 0.25 * 100:
        return cols[1]
    elif val <= 0.5 * 100:
        return cols[2]
    elif val <= 0.75 * 100:
        return cols[3]
    else:
        return cols[4]

def print_map(G, values, prefix, idx, original=True, print_points=False, second_color_scale=False):

    ec = [select_color(v, original, second_color_scale) for v in values]
    fig, ax = ox.plot_graph(G, show=False, close=False, node_size=0, edge_color=ec, bgcolor='#dcdcdc', edge_linewidth=1)
    
    #plt.show()
    
    pattern = defines.OUTPUT_BASE_PATH + '{}_{}.png'
    filename = pattern.format(idx, prefix)
    
    if print_points:
        for _, edge in ox.graph_to_gdfs(G, nodes=False).fillna('').iterrows():
            c = edge['geometry'].centroid
            text = edge['tfcId']
            if text != 0:
    #            ax.annotate(text, (c.x, c.y), c='w')
                ax.scatter(c.x, c.y, c='red')
    
    # for _, edge in ox.graph_to_gdfs(G, nodes=False).fillna('').iterrows():
    #     c = edge['geometry'].centroid
    #     text = edge['tfcId']
    #     if text != 0:
    #         ax.annotate(text, (c.x, c.y), c='w')

    plt.savefig(filename)
    plt.close(fig)
    
    