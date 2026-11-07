import matplotlib.pyplot as plt
import numpy as np

def generate_stomp_comparison(
    n_steps: int = 20, 
    noise_std: float = 0.3,
    high_coeff: float = 0.01,  # scale the high pass noise to better visualize and compare the result.
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate the STOMP comparison data.
    """
    # 1. Define time steps and initial trajectory (smooth sine wave).
    np.random.seed(42)
    x: np.ndarray = np.linspace(0, 2 * np.pi, n_steps)
    initial_trajectory: np.ndarray = 0.5 * np.sin(x) + 0.5

    # 2. Generate completely independent white noise samples.
    # Start and end points are usually fixed (noise is 0).
    white_noise: np.ndarray = np.random.normal(0, noise_std, n_steps)
    white_noise[0] = 0.0
    white_noise[-1] = 0.0
    independent_sample: np.ndarray = initial_trajectory + white_noise

    # 3. Construct the acceleration finite difference matrix K (size n_steps x n_steps).
    K: np.ndarray = np.zeros((n_steps, n_steps))
    for i in range(1, n_steps - 1):
        K[i, i - 1] = 1.0
        K[i, i] = -2.0
        K[i, i + 1] = 1.0
    # Consider the coefficient dt.
    K *= 1 / (2*np.pi / n_steps)**2

    # 4. Compute the smoothing cost matrix R = K^T * K.
    R: np.ndarray = np.dot(K.T, K)

    # 5. Handle boundary conditions to solve for the covariance matrix.
    # Since the start and end points are fixed, the first and last rows/columns of R are all 0,
    # which makes the matrix singular (non-invertible).
    # Approach: Only invert the submatrix of waypoints that need optimization/noise.
    R_inner: np.ndarray = R[1:-1, 1:-1]

    # Introduce regularization (to prevent numerical instability) and then invert to get the inner covariance matrix.
    epsilon_reg: float = 1e-6
    Sigma_inner: np.ndarray = np.linalg.inv(R_inner + epsilon_reg * np.eye(n_steps - 2))

    # 6. Use Cholesky decomposition to generate naturally smooth low-pass noise.
    # L_inner_lowpass is the lower triangular filtering matrix used to generate low-pass noise.
    L_inner_lowpass: np.ndarray = np.linalg.cholesky(Sigma_inner)
    # L_inner_highpass is the inverse of the lower triangular filtering matrix used to generate high-pass noise.
    L_inner_highpass: np.ndarray = np.linalg.cholesky(np.linalg.inv(Sigma_inner))

    # Generate standard white noise vector.
    z_inner: np.ndarray = np.random.normal(0, noise_std, n_steps - 2)

    # Perform low-pass and high-pass filtering through temporal convolution (matrix multiplication).
    stomp_noise_inner_lowpass: np.ndarray = np.dot(L_inner_lowpass, z_inner)
    stomp_noise_inner_highpass: np.ndarray = np.dot(L_inner_highpass, z_inner)

    # 7. Concatenate boundaries to form the complete STOMP sampling trajectory (purple line effect in the image).
    stomp_noise_lowpass: np.ndarray = np.zeros(n_steps)
    stomp_noise_lowpass[1:-1] = stomp_noise_inner_lowpass
    stomp_sample_lowpass: np.ndarray = initial_trajectory + stomp_noise_lowpass
    stomp_noise_highpass: np.ndarray = np.zeros(n_steps)
    stomp_noise_highpass[1:-1] = stomp_noise_inner_highpass
    stomp_sample_highpass: np.ndarray = initial_trajectory + stomp_noise_highpass * high_coeff
    return x, initial_trajectory, independent_sample, stomp_sample_lowpass, stomp_sample_highpass


def plot_trajectories(
    x: np.ndarray, 
    initial: np.ndarray, 
    independent: np.ndarray, 
    stomp: np.ndarray,
    stomp_inverse: np.ndarray
) -> None:
    plt.figure(figsize=(10, 6))
    
    plt.plot(x, independent, color='#d97706', linewidth=2, label='Independent White Noise')
    plt.plot(x, initial, color='#2563eb', linewidth=2, label='Initial Mean Trajectory')
    plt.plot(x, stomp, color='#7c3aed', linewidth=2, label='STOMP Covariance Sample (lowpass)')
    plt.plot(x, stomp_inverse, color='#16a34a', linewidth=2, label='STOMP Inverse Covariance Sample (highpass)')

    plt.title('Impact of STOMP Covariance Matrix on Trajectory Sampling', fontsize=12, pad=15)
    plt.xlabel('Time Step (rad)', fontsize=10)
    plt.ylabel('Joint Position (rad)', fontsize=10)
    plt.grid(True, linestyle='-', alpha=0.3)
    plt.legend(bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    x, initial, independent, stomp, stomp_inverse = generate_stomp_comparison(n_steps=20, noise_std=0.3)
    plot_trajectories(x, initial, independent, stomp, stomp_inverse)

