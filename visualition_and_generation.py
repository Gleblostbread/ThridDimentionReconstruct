from typing import List, Tuple
from numpy.typing import NDArray
from datetime import datetime
import plotly.graph_objects as go
import numpy as np
import os


def create_planes_mesh(normal: NDArray, basis_1: NDArray, basis_2: NDArray, plane_bias: np.float64, size: float = 2, resolution: int = 10):
    """
    Генерирует координаты сетки для визуализации плоскости в 3D пространстве.

    Плоскость задается точкой привязки (проекцией начала координат на плоскость со смещением)
    и двумя базисными векторами, лежащими в этой плоскости. Функция создает прямоугольную сетку
    точек вокруг точки привязки для использования в графиках Plotly (go.Surface).

    Args:
        normal (NDArray): Нормаль к плоскости (используется только для расчета точки p0).
        basis_1 (NDArray): Первый базисный вектор плоскости (ось U локальной СК).
        basis_2 (NDArray): Второй базисный вектор плоскости (ось V локальной СК).
        plane_bias (np.float64): Смещение плоскости вдоль нормали от начала координат.
                                Точка на плоскости вычисляется как p0 = normal * plane_bias.
        size (float, optional): Половина длины стороны квадратной области визуализации 
                                в локальных координатах плоскости. По умолчанию 2.
        resolution (int, optional): Количество точек по каждой оси сетки. По умолчанию 10.

    Returns:
        Tuple[NDArray, NDArray, NDArray]: Кортеж из трех массивов (X, Y, Z) размерности 
                                        (resolution x resolution), содержащих глобальные 
                                        координаты точек сетки плоскости.
    """
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
        """
        Инициализирует генератор синтетических данных и визуализатор сцены.

        Генерирует случайное облако 3D точек из многомерного нормального распределения
        и проецирует его на заданное количество случайно ориентированных плоскостей.
        Опционально запускает визуализацию и сохранение данных.

        Args:
            points_count (int): Количество генерируемых 3D точек.
            cov_matrix (NDArray): Ковариационная матрица размера 3x3 для генерации точек.
                                Определяет форму и ориентацию облака точек.
            planes_count (int): Количество проекционных плоскостей для генерации.
            auto_visualize (bool, optional): Если True, автоматически отображает интерактивный 
                                            график после инициализации. По умолчанию False.
            save (bool, optional): Если True, сохраняет сгенерированные данные и график 
                                в файлы. По умолчанию False.
            save_folder (str | None, optional): Путь к папке для сохранения данных. 
                                                Если None, создается новая папка в директории 
                                                'data/experiment_{timestamp}'.

        Attributes:
            points (NDArray): Сгенерированное облако точек формы (N, 3).
            planes_count (int): Количество плоскостей.
            normal (List[NDArray]): Список нормалей для каждой плоскости.
            basis (List[Tuple[NDArray, NDArray]]): Список кортежей базисных векторов (u, v) для каждую плоскость.
            plain_bias (List[np.float64]): Список смещений плоскостей от начала координат.
            projections (List[NDArray]): Список 3D координат проекций точек на каждую плоскость.
            coords_2d (List[NDArray]): Список 2D координат проекций в локальном базисе каждой плоскости.
            fig (go.Figure): Объект фигуры Plotly для визуализации.
        """
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
        """
        Генерирует параметры случайной проекционной плоскости.

        Создает случайную нормаль, вычисляет ортонормированный базис плоскости и определяет
        смещение плоскости так, чтобы она находилась на определенном расстоянии от облака точек.

        Args:
            pos (int, optional): Множитель направления смещения плоскости (1 или -1). 
                                Используется для чередования положения плоскостей по разные 
                                стороны от облака точек. По умолчанию 1.

        Returns:
            Tuple[NDArray, NDArray, NDArray, np.float64]: Кортеж, содержащий:
                - normal: Единичный вектор нормали к плоскости.
                - basis_1: Первый единичный базисный вектор, лежащий в плоскости.
                - basis_2: Второй единичный базисный вектор, лежащий в плоскости (ортогонален basis_1).
                - plane_bias: Скалярное смещение плоскости вдоль нормали от начала координат.
        """
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
        """
        Вычисляет ортогональные проекции точек на заданную плоскость.

        Проецирует все точки облака на плоскость, заданную нормалью и смещением, а затем
        преобразует полученные 3D координаты проекций в 2D координаты в локальном базисе плоскости.

        Args:
            normal (NDArray): Вектор нормали к плоскости проекции.
            basis_1 (NDArray): Первый базисный вектор локальной системы координат плоскости.
            basis_2 (NDArray): Второй базисный вектор локальной системы координат плоскости.
            plane_bias (np.float64): Смещение плоскости вдоль нормали.

        Returns:
            Tuple[NDArray, NDArray]: Кортеж из двух массивов:
                - projections: 3D координаты проекций точек на плоскость формы (N, 3).
                - coords_2d: 2D координаты проекций в базисе (basis_1, basis_2) формы (N, 2).
        """
        projections = self.points - (np.sum(normal*self.points, axis=1) - plane_bias).reshape((self.points.shape[0], 1))*normal
        coords_2d = np.column_stack((np.sum(basis_1*projections, axis=1), np.sum(basis_2*projections, axis=1)))
        return projections, coords_2d


    def visualize_points(self):
        """
        Добавляет на график исходное облако 3D точек.

        Точки отображаются синими маркерами с полупрозрачностью.
        """
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
        """
        Добавляет на график поверхности проекционных плоскостей.

        Для каждой плоскости генерируется сетка координат и добавляется как полупрозрачная 
        зеленая поверхность (go.Surface) для визуального обозначения границ проекции.
        """
        for ind in range(self.planes_count):
            X, Y, Z = create_planes_mesh(self.normal[ind], self.basis[ind][0], self.basis[ind][1], self.plain_bias[ind], size = 10, resolution = 10)
            self.fig.add_trace(go.Surface(x=X, y=Y, z=Z, opacity=0.25, colorscale='Greens', 
                showscale=False, name=f'Плоскость {ind+1}', hoverinfo='skip'))

    def visualize_projections(self):
        """
        Добавляет на график точки проекций облака на каждую плоскость.

        Проекции отображаются красными маркерами непосредственно на соответствующих плоскостях.
        """ 
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
        """
        Добавляет на график векторы локального базиса для каждой плоскости.

        Для каждой плоскости отрисовываются два вектора (U и V), начинающиеся в точке 
        привязки плоскости (normal * bias). Вектор U отображается желтым цветом, 
        вектор V — зеленым. Позволяет визуально оценить ориентацию локальных систем координат.

        Args:
            scale (float, optional): Длина отображаемых векторов. По умолчанию 1.5.
            color_u (str, optional): Цвет вектора U (basis_1). По умолчанию 'yellow'.
            color_v (str, optional): Цвет вектора V (basis_2). По умолчанию 'green'.
        """ 
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
        """
        Выполняет полную визуализацию сцены.

        Последовательно вызывает методы отрисовки точек, плоскостей, проекций и базисных векторов,
        а затем отображает итоговый интерактивный график.
        """
        self.visualize_points()
        self.visualize_planes()
        self.visualize_projections()
        self.visualize_basis_vectors()
        self.fig.show()

    def save_data_and_graph(self):
        """
        Сохраняет сгенерированные данные и интерактивный график в файлы.

        Создает директорию (если не указана явно) и сохраняет:
        - Интерактивный HTML-график (plot.html).
        - Исходные 3D точки (points.npy).
        - 2D координаты проекций (coords_2d.npy).
        - Нормали плоскостей (normal.npy).
        - Базисы плоскостей (basis.npy).
        - Смещения плоскостей (plain_bias.npy).
        - 3D координаты проекций (projections.npy).

        Имя папки формируется на основе текущей даты и времени, если save_folder не был 
        указан при инициализации.
        """
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
