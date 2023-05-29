import pandas as pd
import networkx as nx
from scipy.spatial import cKDTree
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from ops.io import read_stack as read
import matplotlib.cm as cm

def initialize_graph(df):
    arr_df = [x for _, x in df.groupby('frame')]
    nodes = df[['frame', 'nucleus_num']].values
    nodes = [tuple(x) for x in nodes]

    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    edges = []
    for df1, df2 in zip(arr_df, arr_df[1:]):
        edges = get_edges(df1, df2)
        G.add_weighted_edges_from(edges)

    return G

def get_edges(df1, df2):
    neighboring_points = 3
    get_label = lambda x: tuple(int(y) for y in x[[2, 3]])

    x1 = df1[['centroid_x', 'centroid_y', 'frame', 'nucleus_num']].values
    x2 = df2[['centroid_x', 'centroid_y', 'frame', 'nucleus_num']].values

    kdt = cKDTree(df1[['centroid_x', 'centroid_y']])
    points = df2[['centroid_x', 'centroid_y']]

    result = kdt.query(points, neighboring_points)
    edges = []
    for i2, (ds, ns) in enumerate(zip(*result)):
        end_node = get_label(x2[i2])
        for d, i1 in zip(ds, ns):
            start_node = get_label(x1[i1])
            w = d
            edges.append((start_node, end_node, w))

    return edges

def analyze_graph(G, cutoff=100, start_frame=1):
    """Trace a path forward from each nucleus in the starting frame. Only keep 
    the paths that reach the final frame.
    """
    start_nodes = [n for n in G.nodes if n[0] == start_frame]
    max_frame = max([frame for frame, _ in G.nodes])

    if len( start_nodes ) < 1: return {}, {}

    cost, path = nx.multi_source_dijkstra(G, start_nodes, cutoff=cutoff)
    cost = {k:v for k,v in cost.items() if k[0] == max_frame}
    path = {k:v for k,v in path.items() if k[0] == max_frame}
    return cost, path

def filter_paths(cost, path, threshold=35):
    """Remove intersecting paths. 
    returns list of one [(frame, label)] per trajectory
    """
    # remove intersecting paths (node in more than one path)
    node_count = Counter(sum(path.values(), []))
    bad = set(k for k,v in node_count.items() if v > 1)
    print('bad', len(bad), len(node_count))

    # remove paths with cost over threshold
    too_costly = [k for k,v in cost.items() if v > threshold]
    bad = bad | set(too_costly)

    relabel = [v for v in path.values() if not (set(v) & bad)]
    assert(len(relabel) > 0)
    return relabel

def filter_paths_better(cost, path, threshold=35):
    """Keep only the best of intersecting paths.
    returns list of one [(frame, label)] per trajectory
    """
    # find intersecting paths (node in more than one path)
    node_count = Counter(sum(path.values(), []))
    bad = set(k for k,v in node_count.items() if v > 1)

    # get all the paths that contain each repeated node
    # keep only the best one (shortest path)
    bad_path_keys = []
    for bad_node in bad:
        paths_to_consider = []
        costs_to_consider = []
        keys_to_consider = []
        for key,p in path.items():
            if bad_node in p:
                paths_to_consider.append( p )
                costs_to_consider.append( cost[key] )
                keys_to_consider.append( key )
        best_path_ind = np.argmin( costs_to_consider )
        for ind, key in enumerate( keys_to_consider ):
            if ind != best_path_ind:
                bad_path_keys.append( key )

    paths_after_removing_duplicates = {}
    for key, p in path.items():
        if key not in bad_path_keys:
            paths_after_removing_duplicates[ key ] = p
            
    node_count_updated = Counter(sum(paths_after_removing_duplicates.values(), []))
    updated_bad = set(k for k,v in node_count_updated.items() if v > 1)
    
    # remove paths with cost over threshold
    too_costly = [k for k,v in cost.items() if v > threshold]
    final_bad = updated_bad | set(too_costly)
    
    #print('final bad', len(final_bad), len(node_count_updated))
    
    relabel = [v for v in paths_after_removing_duplicates.values() if not (set(v) & final_bad)]
    #assert(len(relabel) > 0)
    return relabel


def get_tracked_df_single_tile( data_allt, timepoints, cutoff=250. ):
    # data_allt is the dataframe with data from all timepoints
    # for a single tile 
    # each timepoint should be specified by "frame"

    filtered_paths_per_timepoint = []
    for time in timepoints[1:]:
        data_timep = data_allt[(data_allt['frame'].isin([time-1,time]))].copy()
        cell_tracking_graph = initialize_graph( data_timep )
        cost, path = analyze_graph( cell_tracking_graph, cutoff=cutoff, start_frame=time-1 )
        filtered_paths = filter_paths_better( cost, path, threshold=cutoff )
        filtered_paths_per_timepoint.append( filtered_paths )
        #print( "Final number of filtered paths:", len(filtered_paths) )
    
    
    # link all the paths
    combined_paths_per_timepoint = [filtered_paths_per_timepoint[0]]
    for ind_t in range(len(filtered_paths_per_timepoint)-1 ):
        paths_t1 = combined_paths_per_timepoint[-1]
        paths_t2 = filtered_paths_per_timepoint[ind_t+1]
        
        combined_paths = []
        for path_1 in paths_t1:
            elt_end = path_1[-1]
            for path_2 in paths_t2:
                elt_start = path_2[0]
                if elt_end == elt_start:
                    combined_path = [x for x in path_1]
                    combined_path.append( path_2[-1] )
                    combined_paths.append( combined_path )
        combined_paths_per_timepoint.append( combined_paths )
        
    final_combined_paths = combined_paths_per_timepoint[-1]


    # add track info to dataframe
    tracked_df = {
        'frame': [],
        'nucleus_num': [],
        'tracked_nucleus_num': []
    }
    for i, track in enumerate( final_combined_paths ):
        # track is a list of [(frame, nucleus_num), (frame, nucleus_num),...]
        for frame, nucleus_num in track:
            tracked_df['frame'].append( frame )
            tracked_df['nucleus_num'].append( nucleus_num )
            tracked_df['tracked_nucleus_num'].append( int(i) )
            
    tracked_df = pd.DataFrame( tracked_df )
    data_allt_tracked = pd.merge( data_allt, tracked_df, on=['frame','nucleus_num'] )

    return data_allt_tracked

def get_tracked_df_all_tiles( data_allt, timepoints, cutoff=250. ):
    list_of_tiles = sorted(set(data_allt['tile'].tolist()))

    tracked_dfs_per_tile = []
    for tile in list_of_tiles:
        print( "TRACKING TILE", tile, flush=True )
        data_allt_tile = data_allt[data_allt['tile']==tile ]
        # get the tracked df for this tile
        tracked_df = get_tracked_df_single_tile( data_allt_tile, timepoints, cutoff )
        tracked_dfs_per_tile.append( tracked_df )

    final_tracked_df_all_tiles = pd.concat( tracked_dfs_per_tile )

    return final_tracked_df_all_tiles

