import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

def draw_graph_without_coordinates(edge_list):
    '''
    Given an edge list, draw the graph without coordinates, i.e.,
    node position is random.
    '''
    # create and populate graph object
    graph = nx.Graph()
    for s,t in edge_list:
        nx.add_path(graph, [s,t])

    # draw graph
    nx.draw(graph, with_labels=True, node_color='orange')

def draw_graph(edge_list, coordinates, alpha = 1., node_color = 'orange', node_size=200,\
               numbers = True, offsetx = 0.05, offsety=0.05, fig = None, path = None):
    '''
    Given an edge list and node coordinates, plots the graph.

    Inputs
    edge_list: list of edges [[0,1], [2,3], ...]
    coordinates: list of node coordinates [[x0,y0], [x1,y1], ...]
    node_color: color to use for drawing nodes
    node_size: size of nodes in the figure
    offsetx/y: offset of the node label in the figure
    '''
    if fig is None:
        fig = plt.figure()

    plt.vlines(-0.5, -0.5,0.5, alpha=0.75, color = 'lightsteelblue', linestyle = '--')
    plt.vlines(0.5, -0.5,0.5, alpha=0.75, color = 'lightsteelblue', linestyle = '--')

    plt.hlines(-0.5, -0.5,0.5, alpha=0.75, color = 'lightsteelblue', linestyle = '--')
    plt.hlines(0.5, -0.5,0.5, alpha=0.75, color = 'lightsteelblue', linestyle = '--')
    for i,j in edge_list:
        # plot edges
        plt.plot([coordinates[i][0], coordinates[j][0]], [coordinates[i][1], coordinates[j][1]], color = 'k', alpha = alpha, zorder=1)
    # plot nodes
    plt.scatter(coordinates[:, 0], coordinates[:, 1],\
                 alpha = alpha, s=node_size, marker = 'o', linewidth=0, color = node_color, zorder=2)
    if numbers == True:
        for i,j in edge_list:
            # plot node labels:
            plt.text(coordinates[i][0]-offsetx, coordinates[i][1]-offsety, i)
            plt.text(coordinates[j][0]-offsetx, coordinates[j][1]-offsety, j)
    plt.xticks([])
    plt.yticks([])
    sns.despine(fig=fig, bottom=True, left=True, top=True, right=True)
    if path is not None:
        fig.savefig(path, bbox_inches='tight')

def draw_linear_fit(xorig, yorig, xfit, yfit, xlabel='', ylabel='', path = None):
    plt.figure()
    plt.plot(xfit, yfit, color = 'k', label = 'fit')
    plt.plot(xorig, yorig, color = 'darkred', linewidth = 0, marker = 'o', markersize=5, label = 'measured')
    plt.ylabel(ylabel, fontsize = 14)
    plt.xlabel(xlabel, fontsize=14)
    plt.legend(fontsize=12, frameon=False)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, bbox_inches='tight')
