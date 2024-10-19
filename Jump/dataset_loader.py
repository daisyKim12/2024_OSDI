import os
import numpy as np

def read_fbin(file_path):
    with open(file_path, 'rb') as f:
        # 벡터의 수와 차원 읽기
        num_vectors = np.fromfile(f, dtype=np.uint32, count=1)[0]
        dimension = np.fromfile(f, dtype=np.uint32, count=1)[0]
        # print(f"Reading fbin: {num_vectors} vectors with dimension {dimension}")
        
        # 벡터 데이터 읽기
        vectors = np.fromfile(f, dtype=np.float32).reshape(num_vectors, dimension)
        
    return vectors

def read_ibin(file_path):
    with open(file_path, 'rb') as f:
        # 벡터의 수와 차원 읽기
        num_vectors = np.fromfile(f, dtype=np.uint32, count=1)[0]
        dimension = np.fromfile(f, dtype=np.uint32, count=1)[0]
        
        # 벡터 데이터 읽기
        vectors = np.fromfile(f, dtype=np.int32).reshape(num_vectors, dimension)
        
    return vectors

def fetch_dataset(base_dir, dataset_name):
    """
    주어진 데이터셋 이름에 해당하는 데이터셋, 쿼리, 정답 이웃, 정답 거리를 불러옵니다.
    
    base_dir: 데이터셋이 저장된 기본 디렉토리
    dataset_name: 불러올 데이터셋 이름
    """
    # 데이터셋과 쿼리가 저장된 기본 디렉토리
    base_dir = os.path.join(base_dir, dataset_name)
    
    # 데이터셋과 쿼리 파일 경로
    dataset_path = os.path.join(base_dir, 'base.fbin')
    query_path = os.path.join(base_dir, 'query.fbin')
    gt_neighbors_path = os.path.join(base_dir, 'groundtruth.neighbors.ibin')
    gt_distances_path = os.path.join(base_dir, 'groundtruth.distances.fbin')

    # 데이터셋 로드
    dataset_data = read_fbin(dataset_path)
    queries_data = read_fbin(query_path)
    gt_neighbors = read_ibin(gt_neighbors_path)
    gt_distances = read_fbin(gt_distances_path)

    return dataset_data, queries_data, gt_neighbors, gt_distances
