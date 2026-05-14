from typing import List, Tuple
from numpy.typing import NDArray
import plotly.graph_objects as go
import numpy as np

class VisAndGen:
    def __init__(self, points_count: int, cov_matrix: NDArray, planes_count: int):
        self.points: NDArray = np.random.multivariate_normal(
            mean = [0, 0, 0],
            cov = cov_matrix,
            size=points_count
        )
        self.normal: List[NDArray] = []
        self.basis: List[Tuple[NDArray, NDArray]] = []
        self.plain_bias: List[np.float64] = []
        self.projections: List[NDArray] = []
        self.coords_2d: List[NDArray] = []

        for i in range(planes_count):
            tmp_normal, tmp_basis_1, tmp_basis_2, tmp_plane_bias = self.gen_planes((-1)**i)
            tmp_projections, tmp_coords_2d = self.calc_projections(tmp_normal, tmp_basis_1, tmp_basis_2, tmp_plane_bias)

            self.normal.append(tmp_normal)
            self.basis.append((tmp_basis_1, tmp_basis_2))
            self.plain_bias.append(tmp_plane_bias)
            self.projections.append(tmp_projections)
            self.coords_2d.append(tmp_coords_2d)


    def gen_planes(self, pos: int = 1) -> Tuple[NDArray, NDArray, NDArray, np.float64]:
        normal = np.random.random(3)
        normal/= np.linalg.norm(normal)

        projections = self.points@normal

        plane_bias = pos*2*np.max(projections) #- 0.1*np.min(projections)

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


def create_planes_mesh(normal: NDArray, basis_1: NDArray, basis_2: NDArray, plane_bias: np.float64, size: float = 2, resolution: int = 10):
    p0 = normal*plane_bias

    s = np.linspace(-size, size, resolution)
    t = np.linspace(-size, size, resolution)
    S, T = np.meshgrid(s, t)
    
    # Точки плоскости: p0 + s*u + t*v
    X = p0[0] + S*basis_1[0] + T*basis_2[0]
    Y = p0[1] + S*basis_1[1] + T*basis_2[1]
    Z = p0[2] + S*basis_1[2] + T*basis_2[2]
    
    return X, Y, Z
    

def add_basis_vectors(fig, point, u, v, scale=1.5, color_u='red', color_v='green'):
    """Добавляет два вектора базиса, исходящих из point"""
    # Вектор u
    fig.add_trace(go.Scatter3d(
        x=[point[0], point[0] + scale*u[0]],
        y=[point[1], point[1] + scale*u[1]],
        z=[point[2], point[2] + scale*u[2]],
        mode='lines+markers',
        line=dict(color=color_u, width=4),
        marker=dict(size=4, color=color_u),
        name='Basis U', showlegend=True
    ))
    # Вектор v
    fig.add_trace(go.Scatter3d(
        x=[point[0], point[0] + scale*v[0]],
        y=[point[1], point[1] + scale*v[1]],
        z=[point[2], point[2] + scale*v[2]],
        mode='lines+markers',
        line=dict(color=color_v, width=4),
        marker=dict(size=4, color=color_v),
        name='Basis V', showlegend=True
    ))



def main():
    points_count = 17
    cov_matrix = np.array([[2,      2,    -0.5],
                           [2,      5,    -3],
                           [-0.5,  -3,     3]])
    planes_count = 2

    vis_and_gen = VisAndGen(points_count, cov_matrix, planes_count)
    
    fig = go.Figure()

    # Общие настройки сцены
    fig.update_layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data',  # Сохранять пропорции осей
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),  # Позиция камеры
                up=dict(x=0, y=0, z=1),  
            )
        ),
        width=800, height=600,
        margin=dict(r=10, l=10, b=10, t=10)
    )

    fig.add_trace(go.Scatter3d(
        x=vis_and_gen.points[:, 0], y=vis_and_gen.points[:, 1], z=vis_and_gen.points[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color='royalblue',      # или массив цветов
            opacity=0.7,
            # colorscale='Viridis',   # если color — массив
            colorbar=dict(title='Value')
        ),
        name='Point Cloud'
    ))

    # Проекции
    fig.add_trace(go.Scatter3d(
        x=vis_and_gen.projections[0][:, 0], y=vis_and_gen.projections[0][:, 1], z=vis_and_gen.projections[0][:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color='red',      # или массив цветов
            opacity=0.7,
            # colorscale='Viridis',   # если color — массив
            colorbar=dict(title='Value')
        ),
        name='Projections'
    ))
    fig.add_trace(go.Scatter3d(
        x=vis_and_gen.projections[1][:, 0], y=vis_and_gen.projections[1][:, 1], z=vis_and_gen.projections[1][:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color='red',      # или массив цветов
            opacity=0.7,
            # colorscale='Viridis',   # если color — массив
            colorbar=dict(title='Value')
        ),
        name='Projections'
    ))

    X, Y, Z = create_planes_mesh(vis_and_gen.normal[0], vis_and_gen.basis[0][0], vis_and_gen.basis[0][1], vis_and_gen.plain_bias[0], size = 10, resolution = 10)
    X2, Y2, Z2 = create_planes_mesh(vis_and_gen.normal[1], vis_and_gen.basis[1][0], vis_and_gen.basis[1][1], vis_and_gen.plain_bias[1], size = 10, resolution = 10)

    # Плоскости
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, opacity=0.25, colorscale='Greens', 
    showscale=False, name='Plane', hoverinfo='skip'))
    fig.add_trace(go.Surface(x=X2, y=Y2, z=Z2, opacity=0.25, colorscale='Greens', 
    showscale=False, name='Plane', hoverinfo='skip'))

    # Базисы
    add_basis_vectors(fig, vis_and_gen.normal[0]*vis_and_gen.plain_bias[0], vis_and_gen.basis[0][0], vis_and_gen.basis[0][1], scale=1.5, color_u='yellow', color_v='green')
    add_basis_vectors(fig, vis_and_gen.normal[1]*vis_and_gen.plain_bias[1], vis_and_gen.basis[1][0], vis_and_gen.basis[1][1], scale=1.5, color_u='yellow', color_v='green')
    fig.show()


if __name__ == "__main__":
    main()
