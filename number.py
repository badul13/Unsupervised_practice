import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans

# 손글씨 숫자 데이터셋 로드
digits = load_digits()
data = digits.data
images = digits.images
labels = digits.target

# 데이터 확인
print("Data shape:", data.shape)
print("Number of images:", len(images))
print("Unique labels:", np.unique(labels))

# 군집 수 범위 설정
K = range(1, 15)
inertia = []

# 엘보우 방법을 사용하여 최적의 군집 수 찾기
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    inertia.append(kmeans.inertia_)

# 엘보우 그래프 그리기
plt.plot(K, inertia, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()

# 두 번째 차분 계산 (관성의 변화율)
diff = np.diff(inertia)
diff2 = np.diff(diff)

# 두 번째 차분에서 최대값의 인덱스 찾기
optimal_k = np.argmax(diff2) + 2  # 두 번째 차분의 인덱스는 2차원으로 변화했기 때문에 +2

print(f"Optimal number of clusters: {optimal_k}")

# 최적의 군집 수로 K-means 클러스터링 수행
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(data)
cluster_labels = kmeans.labels_

# 클러스터링 결과의 혼동 행렬 시각화
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 혼동 행렬 생성
conf_matrix = confusion_matrix(labels, cluster_labels)

# 혼동 행렬 시각화
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.title("Confusion Matrix")
plt.xlabel("Cluster Label")
plt.ylabel("True Label")
plt.show()

# 각 클러스터에 속한 데이터 포인트 수 확인
for i in range(optimal_k):
    cluster = data[cluster_labels == i]
    print(f"Cluster {i} contains {len(cluster)} points")

# 각 클러스터의 일부 이미지를 시각화
fig, axs = plt.subplots(optimal_k, 10, figsize=(15, 15))
for i in range(optimal_k):
    cluster_indices = np.where(cluster_labels == i)[0]
    for j in range(10):
        if j < len(cluster_indices):
            axs[i, j].imshow(images[cluster_indices[j]], cmap='gray')
            axs[i, j].axis('off')
        else:
            axs[i, j].axis('off')
plt.show()
