import numpy as np
from scipy.spatial import procrustes

def perform_icp(source_array, target_array, max_iterations=100):
    # Ensure that the arrays have the same number of points
    assert source_array.shape == target_array.shape, "Arrays must have the same shape."

    # Perform ICP registration using scipy's procrustes function
    _, _, transformation = procrustes(target_array, source_array)

    # Construct the 4x4 transformation matrix
    transformation_matrix = np.identity(4)
    print(transformation_matrix)
    transformation_matrix[:3, :3] = transformation['rotation']
    transformation_matrix[:3, 3] = transformation['translation']

    # Apply the transformation matrix to the source array
    homogeneous_source_array = np.hstack((source_array, np.ones((source_array.shape[0], 1))))
    transformed_array_homogeneous = np.dot(transformation_matrix, homogeneous_source_array.T).T
    transformed_array = transformed_array_homogeneous[:, :3]

    return transformed_array

if __name__ == "__main__":
    # Example arrays with multiple points
    source_array = np.array([[1.0, 2.0, 3.0],
                             [4.0, 5.0, 6.0],
                             [7.0, 8.0, 9.0]])

    target_array = np.array([[2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0],
                             [8.0, 9.0, 10.0]])

    # Perform ICP registration
    transformed_array = perform_icp(source_array, target_array)

    # Print the transformed array
    print("Original array:")
    print(source_array)
    print("Transformed array:")
    print(transformed_array)
