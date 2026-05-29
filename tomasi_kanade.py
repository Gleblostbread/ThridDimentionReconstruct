from numpy.typing import NDArray
from os import PathLike
import numpy as np


FilePath = str | PathLike


class TomasiKanade:
    def __init__(self, data: FilePath | NDArray[np.float64], true_coords: FilePath | NDArray[np.float64] | None = None):
        """
        Инициализирует реконструктор структуры Томази-Канаде.

        Загружает входные данные (2D проекции) и, опционально, истинные 3D координаты
        для последующей верификации. Данные могут быть переданы как путь к файлу .npy
        или как объект NumPy ndarray.

        Args:
            data (FilePath | NDArray[np.float64]): 
                Двумерные координаты проекций точек. 
                Если str/Path: путь к файлу .npy с массивом формы (F, N, 2), 
                где F — количество плоскостей, N — количество точек.
                Если NDArray: непосредственно массив numpy указанной формы.
            
            true_coords (FilePath | NDArray[np.float64] | None, optional): 
                Истинные трехмерные координаты точек для сравнения.
                Если str/Path: путь к файлу .npy с массивом формы (N, 3).
                Если NDArray: непосредственно массив numpy.
                По умолчанию None (если эталонные данные отсутствуют).

        Raises:
            TypeError: Если тип переданных аргументов data или true_coords не соответствует
                    ожидаемым (str, PathLike или np.ndarray).

        Attributes:
            coords_2d (NDArray): Загруженные 2D проекции.
            true_coords (NDArray | None): Загруженные истинные координаты или None.
            M_aff (NDArray | None): Матрица аффинных параметров проекций (2F x 3).
            affine_coordinates (NDArray | None): Аффинно восстановленные координаты точек (N x 3).
            M_metric (NDArray | None): Матрица метрических параметров проекций (2F x 3).
            metric_coordinates (NDArray | None): Метрически восстановленные координаты точек (N x 3).
            aligned_coordinates (NDArray | None): Координаты после финального выравнивания (Procrustes).
        """
        if isinstance(data, (str, PathLike)):
            self.coords_2d = np.load(data)

        elif isinstance(data, np.ndarray):
            self.coords_2d = data

        else:
            raise TypeError(
                f"Expected data to be a file path (str/Path) or np.ndarray, "
                f"got {type(data).__name__}"
            )
        if true_coords is not None:
            if isinstance(true_coords, (str, PathLike)):
                self.true_coords = np.load(true_coords)

            elif isinstance(true_coords, np.ndarray):
                self.true_coords = true_coords

            else:
                raise TypeError(
                    f"Expected true_coords to be a file path (str/Path) or np.ndarray, "
                    f"got {type(true_coords).__name__}"
                )
        else:
            self.true_coords = None

        self.M_aff: NDArray | None = None
        self.affine_coordinates: NDArray | None = None
        self.M_metric: NDArray | None = None
        self.metric_coordinates: NDArray | None = None
        self.aligned_coordinates: NDArray | None = None

    
    def affine_3d_coordinates(self) -> NDArray[np.float64]:
        """
        Выполняет аффинную реконструкцию 3D структуры методом факторизации измерительной матрицы.

        Реализует первый этап алгоритма Томази-Канаде:
        1. Центрирование 2D проекций по каждой плоскости отдельно.
        2. Сборка измерительной матрицы W размерности (2F x N).
        3. Сингулярное разложение (SVD) матрицы W и усечение до ранга 3.
        4. Разложение усеченной матрицы на произведение матрицы движения (M_aff) 
        и матрицы структуры (S_aff).

        Returns:
            NDArray[np.float64]: Матрица аффинных координат точек S_aff формы (N, 3).
                                Точки восстановлены с точностью до аффинного преобразования.

        Note:
            Результаты сохраняются в атрибуты self.M_aff и self.affine_coordinates.
            Для получения евклидовых координат необходимо вызвать метод metric_upgrade().
        """
        coords_2d = self.coords_2d - np.mean(self.coords_2d, axis=1, keepdims=True)

        # Сборка W
        W = coords_2d.transpose(0, 2, 1).reshape(2 * coords_2d.shape[0], -1)

        # SVD и ранг-3 аппроксимация
        U, s, Vt = np.linalg.svd(W, full_matrices=False)
        k = 3
        U_3, s_k, Vt_3 = U[:, :k], s[:k], Vt[:k, :]
        S_3 = np.diag(np.sqrt(s_k))

        # Аффинная факторизация
        M_aff = U_3 @ S_3
        S_aff = S_3 @ Vt_3
        
        self.M_aff = M_aff
        self.affine_coordinates = S_aff
        return S_aff
    

    def metric_upgrade(self, s_aff: NDArray[np.float64] | None = None, m_aff: NDArray[np.float64] | None = None, er: bool = False) -> NDArray[np.float64]:
        """
        Выполняет метрический апгрейд аффинной реконструкции до евклидовой (с точностью до подобия).

        Реализует второй этап алгоритма Томази-Канаде:
        1. Формирование системы линейных уравнений на основе условий ортонормированности 
        базисов проекционных плоскостей (ортогональность осей и равенство их норм).
        2. Решение однородной системы для нахождения симметричной матрицы A = Q * Q^T.
        3. Извлечение матрицы преобразования Q через спектральное разложение A.
        4. Применение Q к аффинным координатам для получения метрических координат.
        5. Нормализация масштаба проекционных матриц к единичной длине.

        Args:
            s_aff (NDArray[np.float64] | None, optional): 
                Матрица аффинных координат. Если None, используется self.affine_coordinates.
            m_aff (NDArray[np.float64] | None, optional): 
                Матрица аффинных параметров проекций. Если None, используется self.M_aff.
            er (bool, optional): 
                Флаг вычисления и вывода проекционной ошибки. 
                Если True, вычисляет ||W - M_metric * S_metric.T|| / ||W|| и печатает результат.
                По умолчанию False.

        Returns:
            NDArray[np.float64]: Матрица метрических координат точек S_metric формы (N, 3).
                                Точки восстановлены с точностью до преобразования подобия 
                                (вращение, масштаб, сдвиг).

        Raises:
            RuntimeError: Если метод вызван до выполнения affine_3d_coordinates() 
                        и не переданы явные аргументы s_aff/m_aff.

        Note:
            Результаты сохраняются в атрибуты self.M_metric и self.metric_coordinates.
        """
        if s_aff is None:
            S_aff = self.affine_coordinates

        else:
            S_aff = s_aff

        if m_aff is None:
            M_aff = self.M_aff

        else:
            M_aff = m_aff

        C_rows = []
        for i in range(0, M_aff.shape[0], 2):
            m1, m2 = M_aff[i], M_aff[i+1]
            # Ортогональность
            C_rows.append([m1[0]*m2[0], m1[1]*m2[1], m1[2]*m2[2],
                        m1[0]*m2[1] + m1[1]*m2[0],
                        m1[0]*m2[2] + m1[2]*m2[0],
                        m1[1]*m2[2] + m1[2]*m2[1]])
            # Равенство норм
            C_rows.append([m1[0]**2 - m2[0]**2, m1[1]**2 - m2[1]**2, m1[2]**2 - m2[2]**2,
                        2*(m1[0]*m1[1] - m2[0]*m2[1]),
                        2*(m1[0]*m1[2] - m2[0]*m2[2]),
                        2*(m1[1]*m1[2] - m2[1]*m2[2])])
            
        C = np.array(C_rows)

        _, _, Vt_C = np.linalg.svd(C, full_matrices=False)
        a = Vt_C[-1, :]

        A = np.array([[a[0], a[3], a[4]],
                    [a[3], a[1], a[5]],
                    [a[4], a[5], a[2]]])

        U_A, S_A, _ = np.linalg.svd(A, full_matrices=True)
        Q = U_A @ np.diag(np.sqrt(np.maximum(S_A, 1e-12)))

        # Применение Q
        S_metric = np.linalg.solve(Q, S_aff).T
        M_metric = M_aff @ Q
        
        # Нормализация проекторов к единичной длине + компенсация масштаба в точках
        norms = np.linalg.norm(M_metric, axis=1)
        avg_norm = np.mean(norms)
        M_metric = M_metric / avg_norm
        S_metric = S_metric * avg_norm
        
        if er:
            coords_2d = self.coords_2d - np.mean(self.coords_2d, axis=0)
            W = coords_2d.transpose(0, 2, 1).reshape(2 * coords_2d.shape[0], -1)
            error = np.linalg.norm(W - M_metric @ S_metric.T) / np.linalg.norm(W)
            print(f"Проекционная ошибка: {error:.2e}")

        self.metric_coordinates = S_metric
        self.M_metric = M_metric
        return S_metric


    def align_via_procrustes_anchors(self, anchor_indices: list[int] | NDArray[np.int64], source: NDArray[np.float64] | None = None, target: NDArray[np.float64] | None = None) -> NDArray[np.float64]:
        """
        Вычисляет преобразование подобия (вращение, масштаб, сдвиг) и применяет его к облаку точек.

        Поддерживает два режима работы:
        1. Выравнивание по эталону (Ground Truth):
        Если передан массив `target`, метод вычисляет оптимальное преобразование,
        совмещающее опорные точки (якоря) из `source` с соответствующими точками из `target`.
        Преобразование вычисляется через алгоритм Кабша-Про́круста с перебором двух вариантов
        вращения (с учетом и без учета отражения) для минимизации ошибки на якорях.
        
        2. Каноническое выравнивание (без эталона):
        Если `target=None`, метод строит локальную правую ортонормированную систему координат
        на основе трех указанных якорей из `source`:
        - Начало координат совпадает с первым якорем.
        - Ось X направлена от первого якоря ко второму.
        - Ось Y лежит в плоскости, образованной тремя якорями.
        - Ось Z дополняет базис до правой тройки.
        Затем вычисляется преобразование, переводящее якоры в эти канонические координаты.

        Args:
            anchor_indices (list[int] | NDArray[np.int64]): 
                Индексы опорных точек в массиве `source`.
                Обязательный параметр. Для канонического выравнивания должен содержать ровно 3 индекса.
                
            source (NDArray[np.float64] | None, optional): 
                Исходное облако точек формы (N, 3). 
                Если None, используется `self.metric_coordinates` (результат метрического апгрейда).
                
            target (NDArray[np.float64] | None, optional): 
                Эталонное облако точек.
                Может быть полным облаком (размер N, тогда используются точки по индексам `anchor_indices`)
                или массивом только якорей (размер len(anchor_indices)).
                Если None, выполняется каноническое выравнивание.

        Returns:
            NDArray[np.float64]: Массив выровненных координат точек формы (N, 3).

        Raises:
            ValueError: Если `source` не указан и `self.metric_coordinates` отсутствует;
                        если для канонического выравнивания указано не ровно 3 индекса;
                        если якоря вырождены (совпадают или лежат на одной прямой);
                        если размерность `target` не согласована с `source` или `anchor_indices`.

        Note:
            Результат сохраняется в атрибут `self.aligned_coordinates`.
            Алгоритм гарантирует минимизацию среднеквадратичной ошибки на опорных точках.
        """
        if source is None:
            if self.metric_coordinates is not None:
                source = self.metric_coordinates
            else:
                raise ValueError("Source points are not provided and metric_coordinates is not set.")

        idx = np.array(anchor_indices)
        
        # Определение целевых точек для расчета (tgt_for_calc)
        if target is not None:
            if len(target) == len(source):
                # Случай 1: Target - полное облако, берем якоря по индексам
                tgt_for_calc = target[idx]
            elif len(target) == len(idx):
                # Случай 2: Target - уже массив якорей, используем как есть
                tgt_for_calc = target
            else:
                raise ValueError(f"Mismatch in target size. Expected {len(source)} or {len(idx)}, got {len(target)}")
            
            src_for_calc = source[idx]
        else:
            # Случай 3: Каноническое выравнивание (нет Ground Truth)
            if len(idx) != 3:
                raise ValueError("Для канонического выравнивания требуется ровно 3 индекса.")
            
            src_anchors = source[idx]
            p1, p2, p3 = src_anchors
            
            v12 = p2 - p1
            v13 = p3 - p1
            
            dist_12 = np.linalg.norm(v12)
            if dist_12 < 1e-9: raise ValueError("Точки 1 и 2 слишком близки.")
            e_x = v12 / dist_12
            
            proj_len = np.dot(v13, e_x)
            v13_orth = v13 - proj_len * e_x
            dist_orth = np.linalg.norm(v13_orth)
            if dist_orth < 1e-9: raise ValueError("Точки коллинеарны.")
            e_y = v13_orth / dist_orth
            
            tgt_p1 = np.array([0.0, 0.0, 0.0])
            tgt_p2 = np.array([dist_12, 0.0, 0.0])
            tgt_p3 = np.array([proj_len, dist_orth, 0.0])
            
            src_for_calc = src_anchors
            tgt_for_calc = np.array([tgt_p1, tgt_p2, tgt_p3])

        # --- Алгоритм Кабша-Про́круста (единый блок) ---
        
        # Центрирование якорей
        cs = src_for_calc.mean(axis=0)
        ct = tgt_for_calc.mean(axis=0)
        
        s_c = src_for_calc - cs
        t_c = tgt_for_calc - ct
        
        # Масштаб (RMS)
        scale_s = np.sqrt(np.mean(np.sum(s_c**2, axis=1)))
        scale_t = np.sqrt(np.mean(np.sum(t_c**2, axis=1)))
        
        if scale_s < 1e-9 or scale_t < 1e-9:
            return source.copy()
            
        s_n = s_c / scale_s
        t_n = t_c / scale_t
        
        # Кросс-ковариация
        H = s_n.T @ t_n
        U, _, Vt = np.linalg.svd(H)
        
        # Вращение
        s_n = s_c / scale_s
        t_n = t_c / scale_t
        
        # Кросс-ковариация
        H = s_n.T @ t_n
        U, _, Vt = np.linalg.svd(H)
        
        # Вариант A: Стандартное решение (может содержать отражение)
        R_a = Vt.T @ U.T
        aligned_a = ((source - cs) / scale_s @ R_a) * scale_t + ct
        
        # Вариант B: Принудительное вращение (det = +1)
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
        R_b = U @ D @ Vt
        aligned_b = ((source - cs) / scale_s @ R_b) * scale_t + ct
        
        # Выбираем вариант с минимальной ошибкой НА якорях
        err_a = np.sqrt(np.mean(np.sum((aligned_a[idx] - tgt_for_calc)**2, axis=1)))
        err_b = np.sqrt(np.mean(np.sum((aligned_b[idx] - tgt_for_calc)**2, axis=1)))
        
        final_aligned = aligned_a if err_a < err_b else aligned_b
        
        self.aligned_coordinates = final_aligned
        return final_aligned

    

    def reconstruct(self, anchor_indices: NDArray[np.float64] | None = None, anchors: NDArray[np.int64] | None = None) -> NDArray:
        """
        Выполняет полный цикл восстановления 3D структуры по 2D проекциям.

        Последовательно вызывает методы:
        1. `affine_3d_coordinates()`: Аффинная реконструкция через SVD измерительной матрицы.
        2. `metric_upgrade()`: Переход от аффинной модели к евклидовой (метрический апгрейд).
        3. `align_via_procrustes_anchors()`: Финальное выравнивание восстановленного облака.

        Args:
            anchor_indices (list[int] | NDArray[np.int64] | None, optional): 
                Индексы опорных точек для финального выравнивания.
                Если None, выравнивание не выполняется, возвращается результат метрического апгрейда.
                
            anchors (NDArray[np.float64] | None, optional): 
                Эталонные координаты для выравнивания (Ground Truth).
                Если указано, используется как целевой массив `target` в методе выравнивания.
                Если None, но задан `self.true_coords`, используется он.
                Если оба значения None, выполняется каноническое выравнивание (по умолчанию для 3 якорей).

        Returns:
            NDArray[np.float64]: 
                Если выполнено выравнивание: массив выровненных координат (N, 3).
                Если выравнивание пропущено (`anchor_indices is None`): массив метрических координат (N, 3).

        Note:
            Метод является высокоуровневым интерфейсом класса.
            Промежуточные результаты сохраняются в атрибутах `self.affine_coordinates`,
            `self.metric_coordinates` и `self.aligned_coordinates`.
        """
        self.affine_3d_coordinates()
        self.metric_upgrade()
        if anchors is None:
            self.align_via_procrustes_anchors(target=self.true_coords, anchor_indices=anchor_indices)
        else:
            self.align_via_procrustes_anchors(target=anchors, anchor_indices=anchor_indices)
        
        return self.aligned_coordinates
