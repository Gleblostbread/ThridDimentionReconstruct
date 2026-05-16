from numpy.typing import NDArray
import plotly.graph_objects as go
import numpy as np
import os


def align_via_procrustes_anchors(source, target, anchor_indices):
    """
    Вычисляет преобразование подобия ТОЛЬКО по якорным точкам,
    затем применяет его ко всему облаку source.
    """
    idx = np.array(anchor_indices)
    src_anchors = source[idx]
    tgt_anchors = target[idx]

    # 1. Центрирование по якорям
    cs = src_anchors.mean(axis=0)
    ct = tgt_anchors.mean(axis=0)
    s_c = src_anchors - cs
    t_c = tgt_anchors - ct
    
    # 2. Масштаб (вычисляется только по якорям)
    scale_s = np.sqrt(np.mean(np.sum(s_c**2, axis=1)))
    scale_t = np.sqrt(np.mean(np.sum(t_c**2, axis=1)))
    if scale_s < 1e-9 or scale_t < 1e-9:
        return source.copy()
        
    s_n = s_c / scale_s
    t_n = t_c / scale_t
    
    # 3. Кросс-ковариация и SVD (только по якорям)
    H = s_n.T @ t_n
    U, _, Vt = np.linalg.svd(H)
    
    # Вариант A: Стандартное решение
    R_a = Vt.T @ U.T
    # Вариант B: Принудительное вращение (det = +1)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R_b = U @ D @ Vt
    
    # 4. Применяем оба варианта ко ВСЕМУ облаку
    # Формула: new = ((old - centroid_src) / scale_src @ R) * scale_tgt + centroid_tgt
    aligned_a = ((source - cs) / scale_s @ R_a) * scale_t + ct
    aligned_b = ((source - cs) / scale_s @ R_b) * scale_t + ct
    
    # 5. Выбираем вариант с минимальной ошибкой ТОЛЬКО на якорях
    err_a = np.sqrt(np.mean(np.sum((aligned_a[idx] - tgt_anchors)**2, axis=1)))
    err_b = np.sqrt(np.mean(np.sum((aligned_b[idx] - tgt_anchors)**2, axis=1)))
    
    return aligned_a if err_a < err_b else aligned_b


def main():
    dir_path = os.path.dirname(__file__)
    data_path = os.path.join(dir_path, 'data', 'experiment_1')
    
    true_points: NDArray = np.load(os.path.join(data_path, 'points.npy'))
    true_points -= np.mean(true_points, axis=0)
    
    coords_2d: NDArray = np.load(os.path.join(data_path, 'coords_2d.npy')) 
    coords_2d -= np.mean(coords_2d, axis=1, keepdims=True)

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
    
    # Сборка матрицы ограничений C
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

    # Whitening для устойчивости SVD
    row_norms = np.linalg.norm(C, axis=1, keepdims=True)
    row_norms[row_norms < 1e-12] = 1.0
    C = C / row_norms

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
    
    error = np.linalg.norm(W - M_metric @ S_metric.T) / np.linalg.norm(W)
    print(f"Проекционная ошибка: {error:.2e}")

    # Выравнивание только по 3 точкам 
    ANCHOR_INDICES = [0, 1, 2] 
    print(f"Вычисление преобразования подобия по якорям: {ANCHOR_INDICES}")
    
    S_aligned_vis = align_via_procrustes_anchors(S_metric, true_points, ANCHOR_INDICES)

    # Дополнительная проверка сохранения геометрии
    def pairwise_dists(P):
        diff = P[:, None, :] - P[None, :, :]
        return np.sqrt(np.sum(diff**2, axis=2))[np.triu_indices(P.shape[0], k=1)]
    
    corr = np.corrcoef(pairwise_dists(true_points), pairwise_dists(S_metric))[0, 1]
    rmse = np.sqrt(np.mean(np.sum((S_aligned_vis - true_points)**2, axis=1)))
    print(f"Корреляция попарных расстояний: {corr:.10f}")
    print(f"RMSE после выравнивания (по всем точкам): {rmse:.2e}")


    fig = go.Figure()
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        width=1000, height=800, title=f"RMSE: {rmse:.2e} (Aligned by 3 Anchors)"
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