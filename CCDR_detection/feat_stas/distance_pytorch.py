import torch
_SCALE = 1000
def gaussian_kernel_matrix_batch(x, y, sigma, batch_size):
    """
    Compute the Gaussian kernel matrix in batches to reduce memory usage.
    """
    n_x, n_y = x.size(0), y.size(0)
    kernel_matrix = torch.zeros((n_x, n_y), device=x.device, dtype=torch.float32)

    for i in range(0, n_x, batch_size):
        for j in range(0, n_y, batch_size):
            batch_x = x[i:i + batch_size]
            batch_y = y[j:j + batch_size]
            dist = torch.cdist(batch_x, batch_y) ** 2
            kernel_matrix[i:i + batch_size, j:j + batch_size] = torch.exp(-dist / (2 * sigma ** 2))

    return kernel_matrix


def mmd(x, y, sigma=10, batch_size=4096):
    """
    Compute the MMD (Maximum Mean Discrepancy) between two samples, x and y, using the Gaussian kernel in batches.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = torch.from_numpy(x).to(device).to(torch.float32), torch.from_numpy(y).to(device).to(torch.float32)

    x_kernel = gaussian_kernel_matrix_batch(x, x, sigma, batch_size=batch_size)
    y_kernel = gaussian_kernel_matrix_batch(y, y, sigma, batch_size=batch_size)
    xy_kernel = gaussian_kernel_matrix_batch(x, y, sigma, batch_size=batch_size)

    mmd = _SCALE * (x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean())
    return mmd.item()

# def gaussian_kernel_matrix(x, y, sigma):
#     """
#     Efficiently compute the Gaussian kernel matrix using broadcasting and torch functions.
#     """
#     x_size = x.size(0)
#     y_size = y.size(0)
#     dim = x.size(1)
#     x = x.unsqueeze(1)  # Shape: (x_size, 1, dim)
#     y = y.unsqueeze(0)  # Shape: (1, y_size, dim)
#     tiled_x = x.expand(x_size, y_size, dim)
#     tiled_y = y.expand(x_size, y_size, dim)
#     kernel_matrix = torch.exp(-torch.sum((tiled_x - tiled_y) ** 2, dim=2) / (2 * sigma ** 2))
#     return kernel_matrix
#
# def mmd(x, y, sigma=10):
#     """
#     Compute the MMD (Maximum Mean Discrepancy) between two samples, x and y, using the Gaussian kernel.
#     The function automatically detects and uses GPU if available.
#     """
#     # Automatically detect and use GPU if available, otherwise fallback to CPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
#
#     x_kernel = gaussian_kernel_matrix(x, x, sigma)
#     y_kernel = gaussian_kernel_matrix(y, y, sigma)
#     xy_kernel = gaussian_kernel_matrix(x, y, sigma)
#     mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
#     return mmd

