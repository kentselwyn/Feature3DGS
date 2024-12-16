import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def hist(score_map):
    # score_map = torch.load("/home/koki/code/cc/feature_3dgs_2/all_data/scene0724_00/A/outputs/test_fea_scoreloss*5/rendering/test/ours_10000/score_tensors/00000_smap.pt")
    flattened_image = score_map.squeeze().flatten().float()

    # breakpoint()

    num_bins = 256
    hist = torch.histc(flattened_image, bins=num_bins, min=flattened_image.min(), max=flattened_image.max())
    cumulative_hist = torch.cumsum(hist, dim=0)

    # Normalize the cumulative histogram
    cumulative_hist_normalized = cumulative_hist / cumulative_hist[-1]

    # Choose the desired percentile (e.g., 95%)
    percentile = 0.97
    threshold_bin = torch.where(cumulative_hist_normalized >= percentile)[0][0]

    # Compute the bin edges
    bin_width = (flattened_image.max() - flattened_image.min()) / num_bins
    threshold_value = flattened_image.min() + threshold_bin * bin_width

    print(threshold_value)



def viz_his(score_map):
    if score_map.dim() == 3:
        score_map = score_map.unsqueeze(0)  # Shape: [1, 1, 640, 640]

    score_map = score_map.float()
    flattened_image = score_map.squeeze().flatten().float()

    num_bins = 256
    min_value = flattened_image.min().item()
    max_value = flattened_image.max().item()

    # Compute the histogram using torch.histc
    hist = torch.histc(flattened_image, bins=num_bins, min=min_value, max=max_value)

    # Convert the histogram to a NumPy array for plotting
    hist_np = hist.cpu().numpy()

    # Compute bin centers for plotting
    bin_edges = torch.linspace(min_value, max_value, steps=num_bins + 1)
    bin_edges_np = bin_edges.cpu().numpy()
    bin_centers = (bin_edges_np[:-1] + bin_edges_np[1:]) / 2

    # Step 2: Plot the non-cumulative histogram using a bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers, hist_np, width=(max_value - min_value) / num_bins, color='blue', alpha=0.7, label='Non-Cumulative Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Non-Cumulative Histogram of Pixel Intensities')
    plt.grid(True)
    plt.legend()

    plt.savefig('non_cumulative_histogram.png', format='png', dpi=300)



def viz_cumu_his(image: torch.Tensor):
    if image.dim() == 3:
        image = image.unsqueeze(0)  # Shape: [1, 1, 640, 640]

    image = image.float()
    # Apply max pooling to find local maxima
    pooled = F.max_pool2d(image, kernel_size=7, stride=1, padding=3)
    local_max = (image == pooled)

    # Flatten the image to compute the
    flattened_image = image.squeeze().flatten().float()

    # Compute 
    num_bins = 256
    min_value = flattened_image.min().item()
    max_value = flattened_image.max().item()
    hist = torch.histc(flattened_image, bins=num_bins, min=min_value, max=max_value)

    # Compute the cumulative histogram
    cumulative_hist = torch.cumsum(hist, dim=0)
    cumulative_hist_normalized = cumulative_hist / cumulative_hist[-1]

    # Prepare data for plotting
    cumulative_hist_np = cumulative_hist_normalized.cpu().numpy()

    # Compute bin edges and centers
    bin_edges = torch.linspace(min_value, max_value, steps=num_bins + 1)
    bin_edges_np = bin_edges.cpu().numpy()
    bin_centers = (bin_edges_np[:-1] + bin_edges_np[1:]) / 2

    # Plot the cumulative
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, cumulative_hist_np, label='Cumulative Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Histogram of Pixel Intensities')
    plt.grid(True)
    plt.legend()

    plt.savefig('histogram_plot.png', format='png', dpi=300)    



if __name__=="__main__":
    score_map = torch.load("/home/koki/code/cc/feature_3dgs_2/all_data/scene0724_00/A/outputs/test_fea_scoreloss*5/rendering/test/ours_10000/score_tensors/00000_smap.pt")
    hist(score_map)
    breakpoint()

