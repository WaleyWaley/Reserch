from torch_geometric.data import HeteroData
import numpy as np
import torch


def create_asymmetric_fuses_edges(window_size, past_radius, future_radius, device):
    """
    【核心修改】创建一个非对称的融合边连接。
    """
    # 1. 为 IMU -> WiFi (因果) 创建边:
    #    WiFi 节点 'i' (目标) 只能看到过去和当前的 IMU 节点 'j'。
    causal_src_nodes, causal_dst_nodes = [], []
    for i in range(window_size):  # 目标时间 (WiFi)
        start = max(0, i - past_radius)
        end = i + 1
        for j in range(start, end):  # 源时间 (IMU)
            causal_src_nodes.append(j)
            causal_dst_nodes.append(i)
    
    edge_index_IMU_to_WiFi = torch.stack([
        torch.tensor(causal_src_nodes, dtype=torch.long, device=device),
        torch.tensor(causal_dst_nodes, dtype=torch.long, device=device)
    ], dim=0)

    # 2. 为 WiFi -> IMU (前瞻) 创建边:
    #    IMU 节点 'i' (目标) 可以看到当前和未来的 WiFi 节点 'j'。
    future_src_nodes, future_dst_nodes = [], []
    for i in range(window_size):  # 目标时间 (IMU)
        start = i
        end = min(window_size, i + future_radius + 1)
        for j in range(start, end):  # 源时间 (WiFi)
            future_src_nodes.append(j)
            future_dst_nodes.append(i)

    edge_index_WiFi_to_IMU = torch.stack([
        torch.tensor(future_src_nodes, dtype=torch.long, device=device),
        torch.tensor(future_dst_nodes, dtype=torch.long, device=device)
    ], dim=0)

    return edge_index_IMU_to_WiFi, edge_index_WiFi_to_IMU



def create_temporal_edges_with_attr(node_features, device):
    """辅助函数：根据时间戳创建时间边和delta_t边特征"""
    #根据节点特征的变化幅度创建时间边和边特征。边特征是连续节点特征向量差的L2范数，捕捉了变化的剧烈程度。
    num_nodes = node_features.size(0)
    if num_nodes < 2:
        # 边和属性都为空
        return torch.empty((2, 0), dtype=torch.long, device=device), \
               torch.empty((0, 1), dtype=torch.float, device=device)

    # 创建从前一个节点指向后一个节点的顺序边
    source_nodes = torch.arange(0, num_nodes - 1, device=device)
    dest_nodes = torch.arange(1, num_nodes, device=device)
    edge_index = torch.stack([source_nodes, dest_nodes], dim=0)
    
    # 计算特征向量差的L2范数
    feature_diff = node_features[1:] - node_features[:-1]
    edge_attr = torch.linalg.norm(feature_diff, dim=1).view(-1, 1)

    return edge_index, edge_attr



def create_graph_list_from_df(df, wifi_cols, imu_cols, windows_size, future_steps, device, past_radius=3, future_radius=20):
    """
    【核心修改】总函数现在接受 past_radius 和 future_radius。
    """
    graph_list = []
    num_samples = len(df) - windows_size - future_steps + 1

    accel_cols = [col for col in imu_cols if 'accelerometer' in col]
    gyro_cols = [col for col in imu_cols if 'gyroscope' in col]
    rss_cols = [col for col in wifi_cols if 'RSSI' in col]

    for i in range(num_samples):
        window_df = df.iloc[i : i + windows_size]
        future_df = df.iloc[i + windows_size : i + windows_size + future_steps]
        
        data = HeteroData()
        
        data['wifi'].x = torch.tensor(window_df[wifi_cols].values, dtype=torch.float, device=device)
        data['imu'].x = torch.tensor(window_df[imu_cols].values, dtype=torch.float, device=device)
        
        data['wifi'].time_step = torch.arange(windows_size, device=device)
        data['imu'].time_step = torch.arange(windows_size, device=device)

        # 时间边 (Temporal Edges) - 保持不变
        wifi_edge_index, wifi_edge_attr = create_temporal_edges_with_attr(data['wifi'].x, device)
        imu_edge_index, imu_edge_attr = create_temporal_edges_with_attr(data['imu'].x, device)
        data[('wifi', 'temporal', 'wifi')].edge_index = wifi_edge_index
        data[('imu', 'temporal', 'imu')].edge_index = imu_edge_index
        data[('wifi', 'temporal', 'wifi')].edge_attr = wifi_edge_attr
        data[('imu', 'temporal', 'imu')].edge_attr = imu_edge_attr

        # 融合边 (Fuses Edges) - 【核心修改】
        edge_index_IMU_to_WiFi, edge_index_WiFi_to_IMU = create_asymmetric_fuses_edges(windows_size, past_radius, future_radius, device)
        
        # 分配新的非对称边
        data[('imu', 'fuses', 'wifi')].edge_index = edge_index_IMU_to_WiFi
        data[('wifi', 'fuses', 'imu')].edge_index = edge_index_WiFi_to_IMU
        
        # --- 边属性的分配逻辑需要相应调整 ---
        # 属性总是来自于源 (source) 节点
        
        # for imu -> fuses -> wifi
        forward_src_indices = edge_index_IMU_to_WiFi[0] # 源是 IMU
        # 运动强度
        motion_intensity = np.linalg.norm(window_df[accel_cols].values, axis=1)
        motion_intensity_tensor = torch.tensor(motion_intensity, dtype=torch.float, device=device)
        # 旋转强度
        rotation_intensity = np.linalg.norm(window_df[gyro_cols].values, axis=1)
        rotation_intensity_tensor = torch.tensor(rotation_intensity, dtype=torch.float, device=device)
        
        fuses_edge_attr_forward = torch.cat([
            motion_intensity_tensor[forward_src_indices].view(-1, 1), 
            rotation_intensity_tensor[forward_src_indices].view(-1, 1)
        ], dim=1)
        data[('imu', 'fuses', 'wifi')].edge_attr = fuses_edge_attr_forward

        # for wifi -> fuses -> imu
        backward_src_indices = edge_index_WiFi_to_IMU[0] # 源是 WiFi
        avg_rssi_per_node = window_df[rss_cols].mean(axis=1).values
        avg_rssi_tensor = torch.tensor(avg_rssi_per_node, dtype=torch.float, device=device)
        fuses_edge_attr_backward = avg_rssi_tensor[backward_src_indices].view(-1, 1)
        data[('wifi', 'fuses', 'imu')].edge_attr = fuses_edge_attr_backward

        # 标签赋值
        labels = future_df[['x_coord', 'y_coord']].values
        data.y = torch.tensor(labels, dtype=torch.float, device=device)
        
        graph_list.append(data)
        
    return graph_list

