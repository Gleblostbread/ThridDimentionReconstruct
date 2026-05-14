import plotly.graph_objects as go
import numpy as np



def gen_planes(points: np.ndarray, pos: int = 1):
    normal = np.random.random(3)
    normal/= np.linalg.norm(normal)

    projections = points@normal

    plane_bias = pos*2*np.max(projections) #- 0.1*np.min(projections)

    if normal[0] < 0.9:
        a = np.array([1, 0, 0])
    else:
        a = np.array([0, 1, 0])


    basis_1 = (a - (a @ normal)*normal)
    basis_1 /= np.linalg.norm(basis_1)

    basis_2 = np.cross(normal, basis_1)

    return normal, basis_1, basis_2, plane_bias


def create_planes_mesh(normal: np.ndarray, basis_1: np.ndarray, basis_2: np.ndarray, plane_bias: np.float64, size: float = 2, resolution: int = 10):
    p0 = normal*plane_bias

    s = np.linspace(-size, size, resolution)
    t = np.linspace(-size, size, resolution)
    S, T = np.meshgrid(s, t)
    
    # Точки плоскости: p0 + s*u + t*v
    X = p0[0] + S*basis_1[0] + T*basis_2[0]
    Y = p0[1] + S*basis_1[1] + T*basis_2[1]
    Z = p0[2] + S*basis_1[2] + T*basis_2[2]
    
    return X, Y, Z


def calc_projections(points:np.ndarray, normal: np.ndarray, basis_1: np.ndarray, basis_2: np.ndarray, plane_bias: np.float64):
    projections = points - (np.sum(normal*points, axis=1) - plane_bias).reshape((points.shape[0], 1))*normal
    coords_2d = np.column_stack((np.sum(basis_1*projections, axis=1), np.sum(basis_2*projections, axis=1)))
    return projections, coords_2d
    

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
    points_true = np.random.multivariate_normal(
        mean = [0, 0, 0],
        cov = [[2, 2, -0.5],
               [2, 5, -3],
               [-0.5, -3, 3]],
        size=100
    )

    normal, basis_1, basis_2, plane_bias = gen_planes(points_true, pos=1)

    X, Y, Z = create_planes_mesh(normal, basis_1, basis_2, plane_bias, size=10)

    projections_1, coords_2d = calc_projections(points_true, normal, basis_1, basis_2, plane_bias)

    print(coords_2d)

    normal_2, basis_21, basis_22, plane_bias_2 = gen_planes(points_true, pos=-1)

    X2, Y2, Z2 = create_planes_mesh(normal_2, basis_21, basis_22, plane_bias_2, size=10)

    projections_2, coords_2d_2 = calc_projections(points_true, normal_2, basis_21, basis_22, plane_bias_2)
    
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
        x=points_true[:, 0], y=points_true[:, 1], z=points_true[:, 2],
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
        x=projections_1[:, 0], y=projections_1[:, 1], z=projections_1[:, 2],
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
        x=projections_2[:, 0], y=projections_2[:, 1], z=projections_2[:, 2],
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

    # Плоскости
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, opacity=0.25, colorscale='Greens', 
    showscale=False, name='Plane', hoverinfo='skip'))
    fig.add_trace(go.Surface(x=X2, y=Y2, z=Z2, opacity=0.25, colorscale='Greens', 
    showscale=False, name='Plane', hoverinfo='skip'))

    # Базисы
    add_basis_vectors(fig, normal*plane_bias, basis_1, basis_2, scale=1.5, color_u='yellow', color_v='green')
    add_basis_vectors(fig, normal_2*plane_bias_2, basis_21, basis_22, scale=1.5, color_u='yellow', color_v='green')
    fig.show()


if __name__ == "__main__":
    main()
