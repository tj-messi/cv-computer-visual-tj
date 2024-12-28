import csv
import sys
import numpy as np
from math import sqrt
import random
from matplotlib import pyplot as plt
import pandas as pd


def load_dataset(filename, separator=','):
    """Load dataset with proper handling of header and separator."""
    data_list = []
    try:
        df = pd.read_csv(filename, sep=separator, header=None)
        data_list = df.values.tolist()
    except Exception as e:
        print("Error loading dataset:", e)
    return data_list


def calculateEucl(vector1, vector2):
    """Calculate Euclidean distance between two points."""
    return sqrt(sum((v1 - v2) ** 2 for v1, v2 in zip(vector1, vector2)))


def make_central_points(dataset, k):
    """Generate k random centroids."""
    attributes_num = len(dataset[0])  # Number of features
    central_points = np.zeros((k, attributes_num))
    for index in range(attributes_num):
        index_min = min([child[index] for child in dataset])
        index_max = max([child[index] for child in dataset])
        index_range = index_max - index_min
        for child in central_points:
            child[index] = index_min + index_range * random.random()
    return central_points


def Kmeans(dataset, k):
    """K-means clustering algorithm."""
    data_num = len(dataset)
    attributes_num = len(dataset[0])
    result = np.zeros((data_num, 2))  # Store cluster index and squared distance
    central_points = make_central_points(dataset, k)
    loop = True
    while loop:
        loop = False
        for i in range(data_num):
            min_dis, min_index = float('inf'), -1
            for j in range(k):
                distance = calculateEucl(dataset[i], central_points[j])
                if distance < min_dis:
                    min_dis, min_index = distance, j
            result[i][1] = min_dis ** 2
            if result[i][0] != min_index:
                result[i][0], loop = min_index, True
        for j in range(k):
            cluster_points = [dataset[i] for i in range(data_num) if result[i][0] == j]
            if cluster_points:
                central_points[j] = np.mean(cluster_points, axis=0)
    return central_points, result


def showElbow(dataset, max_centers):
    """Plot the elbow method curve."""
    k_distance = []
    for k in range(1, max_centers + 1):
        central_points, result = Kmeans(dataset, k)
        total_distance = sum(result[i][1] for i in range(len(result)))
        k_distance.append(total_distance)
    plt.plot(range(1, max_centers + 1), k_distance, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum of Squared Distances')
    plt.savefig("Elbow_Method.png")
    plt.show()


def BiKmeans(dataset, k, judge=False):
    """Binary K-means clustering algorithm."""
    data_num = len(dataset)
    record = np.zeros((data_num, 2))  # Store cluster index and squared distance
    central_points = [np.mean(dataset, axis=0).tolist()]  # Start with one centroid
    for i in range(data_num):
        record[i][1] = calculateEucl(central_points[0], dataset[i]) ** 2  # Initial distances

    while len(central_points) < k:
        if judge:
            print(f"Current number of centroids: {len(central_points)}. Visualization step...")
        min_sse = float('inf')
        best_split_idx = -1
        best_new_centers = None
        best_new_result = None

        for i in range(len(central_points)):
            cluster_points = [dataset[j] for j in range(data_num) if record[j][0] == i]
            if not cluster_points:
                continue
            new_centers, new_result = Kmeans(cluster_points, 2)
            sse_split = sum(new_result[:, 1])
            sse_non_split = sum(record[j][1] for j in range(data_num) if record[j][0] != i)
            if sse_split + sse_non_split < min_sse:
                best_split_idx = i
                min_sse = sse_split + sse_non_split
                best_new_centers = new_centers
                best_new_result = new_result.copy()

        for j in range(len(best_new_result)):
            if best_new_result[j][0] == 1:
                best_new_result[j][0] = len(central_points)
            best_new_result[j][0] = best_split_idx if best_new_result[j][0] == 0 else best_new_result[j][0]

        central_points[best_split_idx] = best_new_centers[0].tolist()
        central_points.append(best_new_centers[1].tolist())
        idx = 0
        for j in range(data_num):
            if record[j][0] == best_split_idx:
                record[j] = best_new_result[idx]
                idx += 1

    return central_points, record


def show_cluster_optimized(dataset, central_points, result, save_name, sample_size=10000):
    """
    Optimized visualization of clustering results.
    Randomly samples a subset of points for display to improve performance.
    """
    num_clusters = len(central_points)
    colors = [
        "red", "green", "blue", "yellow", "orange", "purple",
        "pink", "gray", "brown", "cyan", "magenta"
    ]

    # Randomly sample points if dataset is large
    if len(dataset) > sample_size:
        sampled_indices = random.sample(range(len(dataset)), sample_size)
        sampled_dataset = [dataset[i] for i in sampled_indices]
        sampled_result = [result[i] for i in sampled_indices]
    else:
        sampled_dataset = dataset
        sampled_result = result

    # Scatter plot for sampled data
    for i, data_point in enumerate(sampled_dataset):
        cluster_idx = int(sampled_result[i][0])  # Cluster assignment
        plt.scatter(data_point[0], data_point[1], color=colors[cluster_idx % len(colors)], s=10)

    # Plot centroids
    for center in central_points:
        plt.scatter(center[0], center[1], color="black", marker="x", s=100, linewidths=3)

    plt.title("Clustering Result (Optimized)")
    plt.savefig(f"{save_name}_optimized.png")
    plt.show()


def main():
    dataset = load_dataset("box3.csv", ',')  # Adjust path as necessary
    k = 3  # Number of clusters
    print("Data loaded successfully.")
    print(f"Dataset size: {len(dataset)} records.")
    print(f"Target number of clusters: {k}")

    #print(dataset)
    #central_points, result = Kmeans(dataset, k)  # Perform KMeans clustering
    #show_cluster_optimized(dataset, central_points, result, 'final_kmeans')  # Show and save final cluster plot

    judge = True  # Do not visualize step-by-step for simplicity
    #showElbow(dataset, 10)  # Display the elbow method plot for optimal k
    central_points, result = Kmeans(dataset, k)  # Perform BiKMeans clustering
    #show_cluster_optimized(dataset, central_points, result, 'final_corrected')  # Show and save final cluster plot
    show_cluster_optimized(dataset, central_points, result, 'final_corrected')

if __name__ == '__main__':
    main()
