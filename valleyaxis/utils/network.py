import networkx as nx


def lines_to_network(lines):
    """lines is goepandas dataframe with linestrings"""
    G = nx.DiGraph()
    for index, line in lines.iterrows():
        start = line.geometry.coords[0]
        end = line.geometry.coords[-1]
        G.add_edge(start, end, streamID=index)
    return G
