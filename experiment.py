from numpy.typing import NDArray
from tomasi_kanade import TomasiKanade
import plotly.graph_objects as go
import numpy as np
import os



def main():
    dir_path = os.path.dirname(__file__)
    data_path = os.path.join(dir_path, 'experiment_1')
    
    true_points: NDArray = np.load(os.path.join(data_path, 'points.npy'))
    
    coords_2d: NDArray = np.load(os.path.join(data_path, 'coords_2d.npy')) 

    ANCHOR_INDICES = np.array([0, 1, 2])
    reconstructer = TomasiKanade(coords_2d, true_points)
    reconstructer.reconstruct(anchors=true_points[ANCHOR_INDICES], anchor_indices=ANCHOR_INDICES)

    S_aligned_vis = reconstructer.aligned_coordinates



    fig = go.Figure()
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        width=1000, height=800, title=f"RMSE: (Aligned by 3 Anchors)"
    )

    fig.add_trace(go.Scatter3d(
        x=true_points[:, 0], y=true_points[:, 1], z=true_points[:, 2],
        mode='markers', marker=dict(size=6, color='blue', opacity=0.5), name='True Points'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=S_aligned_vis[:, 0], y=S_aligned_vis[:, 1], z=S_aligned_vis[:, 2],
        mode='markers', marker=dict(size=4, color='red', opacity=0.9), name='Recovered'
    ))

    # Подсветим якорные точки
    fig.add_trace(go.Scatter3d(
        x=S_aligned_vis[ANCHOR_INDICES, 0], 
        y=S_aligned_vis[ANCHOR_INDICES, 1], 
        z=S_aligned_vis[ANCHOR_INDICES, 2],
        mode='markers', marker=dict(size=8, color='green', symbol='diamond'), name='Anchors'
    ))

    fig.show()

if __name__ == '__main__':
    main()