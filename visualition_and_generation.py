from typing import List, Tuple
from numpy.typing import NDArray
from datetime import datetime
import plotly.graph_objects as go
import numpy as np
import os


def create_planes_mesh(normal: NDArray, basis_1: NDArray, basis_2: NDArray, plane_bias: np.float64, size: float = 2, resolution: int = 10):
    p0 = normal*plane_bias

    s = np.linspace(-size, size, resolution)
    t = np.linspace(-size, size, resolution)
    S, T = np.meshgrid(s, t)
    
    X = p0[0] + S*basis_1[0] + T*basis_2[0]
    Y = p0[1] + S*basis_1[1] + T*basis_2[1]
    Z = p0[2] + S*basis_1[2] + T*basis_2[2]
    
    return X, Y, Z


class VisAndGen:
    def __init__(self, points_count: int, cov_matrix: NDArray, planes_count: int, 
                 auto_visualize: bool = False, save: bool = False, save_folder: str | None = None):
        self.points: NDArray = np.random.multivariate_normal(
            mean = [0, 0, 0],
            cov = cov_matrix,
            size=points_count
        )
        self.planes_count = planes_count
        self.normal: List[NDArray] = []
        self.basis: List[Tuple[NDArray, NDArray]] = []
        self.plain_bias: List[np.float64] = []
        self.projections: List[NDArray] = []
        self.coords_2d: List[NDArray] = []
        self.save_folder = save_folder

        for i in range(planes_count):
            tmp_normal, tmp_basis_1, tmp_basis_2, tmp_plane_bias = self.gen_planes((-1)**i)
            tmp_projections, tmp_coords_2d = self.calc_projections(tmp_normal, tmp_basis_1, tmp_basis_2, tmp_plane_bias)

            self.normal.append(tmp_normal)
            self.basis.append((tmp_basis_1, tmp_basis_2))
            self.plain_bias.append(tmp_plane_bias)
            self.projections.append(tmp_projections)
            self.coords_2d.append(tmp_coords_2d)

        self.fig = go.Figure()

        self.fig.update_layout(
            scene=dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                aspectmode='data', 
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),  
                    up=dict(x=0, y=0, z=1),  
                )
            ),
            width=1920, height=1080,
            margin=dict(r=10, l=10, b=10, t=10)
        )

        if auto_visualize:
            self.visualize()
        
        if save:
            self.save_data_and_graph()

    def gen_planes(self, pos: int = 1) -> Tuple[NDArray, NDArray, NDArray, np.float64]:
        normal = 2*np.random.random(3) - 1
        normal/= np.linalg.norm(normal)

        projections = self.points@normal
        plane_bias = pos*5*np.max(projections) 

        if normal[0] < 0.9:
            a = np.array([1, 0, 0])
        else:
            a = np.array([0, 1, 0])

        basis_1 = (a - (a @ normal)*normal)
        basis_1 /= np.linalg.norm(basis_1)

        basis_2 = np.cross(normal, basis_1)

        return normal, basis_1, basis_2, plane_bias
    
    def calc_projections(self, normal: NDArray, basis_1: NDArray, basis_2: NDArray, plane_bias: np.float64) -> Tuple[NDArray, NDArray]:
        projections = self.points - (np.sum(normal*self.points, axis=1) - plane_bias).reshape((self.points.shape[0], 1))*normal
        coords_2d = np.column_stack((np.sum(basis_1*projections, axis=1), np.sum(basis_2*projections, axis=1)))
        return projections, coords_2d


    def visualize_points(self):
        self.fig.add_trace(go.Scatter3d(
            x=self.points[:, 0], y=self.points[:, 1], z=self.points[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color='royalblue',
                opacity=0.7,
                colorbar=dict(title='Value')
            ),
            name='Облако точек'
        ))

    
    def visualize_planes(self):
        for ind in range(self.planes_count):
            X, Y, Z = create_planes_mesh(self.normal[ind], self.basis[ind][0], self.basis[ind][1], self.plain_bias[ind], size = 10, resolution = 10)
            self.fig.add_trace(go.Surface(x=X, y=Y, z=Z, opacity=0.25, colorscale='Greens', 
                showscale=False, name=f'Плоскость {ind+1}', hoverinfo='skip'))

    def visualize_projections(self):
        for ind in range(self.planes_count):
            self.fig.add_trace(go.Scatter3d(
                x=self.projections[ind][:, 0], y=self.projections[ind][:, 1], z=self.projections[ind][:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color='red', 
                    opacity=0.7,
                    colorbar=dict(title=f'Проекция на {ind+1} плоскоть')
                ),
                name=f'Проекция на {ind+1} плоскоть'
            ))

    def visualize_basis_vectors(self, scale=1.5, color_u='yellow', color_v='green'):
        for ind in range(self.planes_count):
            point = self.normal[ind]*self.plain_bias[ind]
            self.fig.add_trace(go.Scatter3d(
                x=[point[0], point[0] + scale*self.basis[ind][0][0]],
                y=[point[1], point[1] + scale*self.basis[ind][0][1]],
                z=[point[2], point[2] + scale*self.basis[ind][0][2]],
                mode='lines+markers',
                line=dict(color=color_u, width=4),
                marker=dict(size=4, color=color_u),
                name=f'Вектор U базиса {ind+1} плоскоти', showlegend=True
            ))
            self.fig.add_trace(go.Scatter3d(
                x=[point[0], point[0] + scale*self.basis[ind][1][0]],
                y=[point[1], point[1] + scale*self.basis[ind][1][1]],
                z=[point[2], point[2] + scale*self.basis[ind][1][2]],
                mode='lines+markers',
                line=dict(color=color_v, width=4),
                marker=dict(size=4, color=color_v),
                name=f'Вектор V базиса {ind+1} плоскоти', showlegend=True
            ))

    def visualize(self):
        self.visualize_points()
        self.visualize_planes()
        self.visualize_projections()
        self.visualize_basis_vectors()
        self.fig.show()

    def save_data_and_graph(self):
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        if self.save_folder is None:
            dir_path = os.path.dirname(os.path.abspath(__file__))
            self.save_folder = os.path.join(dir_path, 'data', f'experiment_{now}')
            os.makedirs(self.save_folder, exist_ok=True)
        
        if self.visualize:
            self.fig.update_layout(scene_camera=dict(
                eye=dict(x=2, y=2, z=1.5),      
                up=dict(x=0, y=0, z=1),          
                center=dict(x=0, y=0, z=0)       
            ))

            self.fig.write_html(
                file=os.path.join(self.save_folder, "plot.html"),
                include_plotlyjs=True,          
                full_html=True,                     
                include_mathjax=False               
            )

        np.save(os.path.join(self.save_folder, "points.npy"), self.points)
        np.save(os.path.join(self.save_folder, "coords_2d.npy"), np.array(self.coords_2d))
        np.save(os.path.join(self.save_folder, "normal.npy"), np.array(self.normal))
        np.save(os.path.join(self.save_folder, "basis.npy"), np.array(self.basis))
        np.save(os.path.join(self.save_folder, "plain_bias.npy"), np.array(self.plain_bias))
        np.save(os.path.join(self.save_folder, "projections.npy"), np.array(self.projections))
        

def main():
    points_count = 17
    cov_matrix = np.array([[2,      2,    -0.5],
                           [2,      5,    -3],
                           [-0.5,  -3,     3]])
    planes_count = 4

    vis_and_gen = VisAndGen(points_count, cov_matrix, planes_count, auto_visualize=True, save=True)


if __name__ == "__main__":
    main()
