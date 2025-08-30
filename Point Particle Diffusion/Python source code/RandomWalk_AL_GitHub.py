# Jake S. Bobowski
# University of British Columbia
# Physics Department
# August 29, 2025
# Point Particle Diffusion/Anderson Localization Simulation

def simulate_clustering(N = 10_000, num_steps = 1000, mu_std = 0.01, total_std = 0.02, M = 50):
    """
    Run a particle simulation with spatially disordered drift field and return metrics.
    
    Parameters:
    - N: number of particles
    - num_steps: number of time steps
    - mu_std: standard deviation of the drift field (disorder strength)
    - M: number of grid cells along one axis (M x M total)
    
    Returns:
    - r2_history: list of ⟨r²⟩ over time
    - nn_history: list of mean nearest neighbor distance over time
    - corr_history: list of divergence × excess density correlation over time
    - main figure, <r^2> vs step, average nearest neighbour distance vs step, correlation coefficient vs step
    """
    
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.colors import hsv_to_rgb
    from scipy.spatial import cKDTree
    from scipy.ndimage import zoom
    import warnings
    from tqdm import tqdm

    if mu_std > total_std:
        raise ValueError("mu_std must be less than or equal to total_std")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # Suppress warning messages.
        
        # Set all particles initially at the center (0, 0) within the [-0.5, 0.5] × [-0.5, 0.5] domain.
        x = np.zeros(N)
        y = np.zeros(N)
        
        # Copy in the initial coordinates.
        x_initial = x.copy() 
        y_initial = y.copy()
        
        # Setup the main figure which will show the evolution of the particle positions
        fig_main = plt.figure(figsize = (10, 10))
        
        # Shared ticks and labels for both ax and ax2
        ticks = np.linspace(-0.5, 0.5, 6)
        tick_labels = np.round(ticks, 2)
        
        gs = fig_main.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], wspace=0.25)  # 2 rows, 2 columns
        ax = fig_main.add_subplot(gs[0, 0])   # Top left: dynamic particle + density
        
        # Top right: bar chart showing the particle density vs height
        ax1 = fig_main.add_subplot(gs[0, 1]) 

        # Bottom left: static bias map and probability gradient
        ax2 = fig_main.add_subplot(gs[1, 0]) 
        ax2.set_aspect('equal')
        ax2.set_xlabel(r"$x$ position")
        ax2.set_ylabel(r"$y$ position")
        ax2.set_xlim(-0.5, 0.5)
        ax2.set_ylim(-0.5, 0.5)
        ax2.set_xticks(ticks)
        ax2.set_yticks(ticks)
        ax2.set_xticklabels(tick_labels)
        ax2.set_yticklabels(tick_labels)

        # We will determine the particle density as a function of the height h from the bottom of the y-range.
        # We will count the particles between -0.5 < y < -0.495, -0.495 < y < -0.49, ..., 0.495 < y < 0.5.
        h = np.linspace(-0.5, 0.5, 20) 
        bin_edges = np.linspace(-0.5, 0.5, 21)
        
        
        # Define grid edges
        mean_count = N / M**2
        norm = TwoSlopeNorm(vcenter = 0, vmin = -N/1e3, vmax = N/1e3)  # Tune the scale
        edges = np.linspace(-0.5, 0.5, M + 1)
        
        # Bias the probabilities by drawing a random mean from a Gaussian probability
        # Negative values indicate a drift left for x and down for y
        mu_x = np.random.normal(0, mu_std, size=(M, M))
        mu_y = np.random.normal(0, mu_std, size=(M, M))

        # Determine the divergence of the bias field
        dy, dx = 1/M, 1/M
        divergence = np.gradient(mu_x, dx, axis=1) + np.gradient(mu_y, dy, axis=0)
        div_field = np.sign(divergence) * np.abs(divergence)

        # Coarsen the divergence field for visualization
        block = 5
        M_block = M // block  # 10 blocks if M = 50
        
        # Reshape divergence into (10, 5, 10, 5)
        div_coarse = divergence.reshape(M_block, block, M_block, block)
        
        # Average over each 5x5 region
        div_blocked = div_coarse.mean(axis=(1, 3))  # shape (10, 10)
        div_rescaled = zoom(div_blocked, block, order=0)  # nearest-neighbor upscale to (50, 50)
        vmaxDiv = np.percentile(np.abs(div_rescaled), 99)  # robust scaling

        # Background heatmap of divergence
        ax2.imshow(div_rescaled, origin='lower', cmap='bwr', extent=(-0.5, 0.5, -0.5, 0.5),
           vmin=-vmaxDiv, vmax=vmaxDiv, interpolation='none', alpha=0.4)
        
        # Overlay sparse quiver field to show directional bias
        step = 2 
        skip = (slice(None, None, step), slice(None, None, step))  # (rows, cols)
        Xc, Yc = np.meshgrid((edges[:-1] + edges[1:]) / 2, (edges[:-1] + edges[1:]) / 2)
        ax2.quiver(Xc[skip], Yc[skip], mu_x[skip], mu_y[skip], angles='xy', scale_units='xy', scale=1,
            width=0.005, headwidth=3, headlength=4, headaxislength=3, color='black', alpha=0.6)

        # Loop that increments time and the particle positions
        r2_history = []
        nn_history = []
        corr_history = []
        for step in tqdm(range(num_steps), desc="Simulating", ncols=80):
            if step == 0:
                ax.clear()
                # Plot the initial positions of the particles.
                ax.scatter(x_initial, y_initial, s = 2, color = 'cyan', label = 'Initial Positions')
                # Dynamically plot the new positions. 
                ax.scatter(x, y, s = 0.5, color = 'orange', label = 'Dynamic Positions')
                ax.legend(loc='upper left')
        
            # Generate step magnitudes from a normal distribution
            ix = np.clip(((x + 0.5) * M).astype(int), 0, M - 1)
            iy = np.clip(((y + 0.5) * M).astype(int), 0, M - 1)
            step_mean_x = mu_x[iy, ix]
            step_mean_y = mu_y[iy, ix]

            # The total stdev is the set by mu_std which sets the drift bias, and
            # diff_std which sets the diffusion stdev.  For total_std to be a constant,
            # regardless of the value of mu_std. One should not have total_std < mu_std.
            diff_std = np.sqrt(total_std**2 - mu_std**2)
            x_step = np.random.normal(loc = step_mean_x, scale = diff_std)
            y_step = np.random.normal(loc = step_mean_y, scale = diff_std)
        
            # If x < -0.5, add 1.  A particle that moves past the left boundary will appear near the right boundary.
            # If x > 0.5, subtract 1.  A particle that moves past the right boundary will appear near the left boundary.
            x += x_step
            x = np.where(x < -0.5, x + 1, x)
            x = np.where(x > 0.5, x - 1, x)

            # If y < -0.5, add 1.  A particle that moves past the bottom boundary will appear near the top boundary.
            # If y > 0.5, subtract 1.  A particle that moves past the top boundary will appear near the bottom boundary.
            y += y_step
            y = np.where(y < -0.5, y + 1, y)
            y = np.where(y > 0.5, y - 1, y)
        
            counts, _, _ = np.histogram2d(y, x, bins = (edges, edges))  # shape (M, M)
            delta = counts - mean_count  # deviations from uniform density
            epsilon = 1e-3  # Small floor to avoid log(0)
            signed_log_delta = np.sign(delta) * np.log10(1 + np.abs(delta) / epsilon)
            vmax = np.max(np.abs(signed_log_delta))  # or hard-code for consistent scaling
        
            coords = np.column_stack((x, y))
            tree = cKDTree(coords)
            dists, _ = tree.query(coords, k=2)  # [0] is self, [1] is nearest neighbor
            mean_nn_distance = np.mean(dists[:, 1])
            nn_history.append(mean_nn_distance)
        
            delta_counts = counts - mean_count
            corr_score = np.sum(divergence * delta_counts) / M**2
            corr_history.append(corr_score)
            
            # Plot cell coloring underneath particles
            ax.clear()
            ax.set_aspect('equal')
            ax.set_xlabel(r"$x$ position")
            ax.set_ylabel(r"$y$ position")
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(-0.5, 0.5)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(tick_labels)
            ax.set_yticklabels(tick_labels)
            ax.imshow(delta, origin = 'lower', cmap = 'bwr',  norm = norm,
              extent = (-0.5, 0.5, -0.5, 0.5), interpolation = 'none', alpha = 0.5)
            ax.scatter(x_initial, y_initial, s = 2, color = 'cyan', label = 'Initial Positions') # Plot the initial positions of the particles.
            ax.scatter(x, y, s = 0.5, color = 'orange', label = 'Dynamic Positions') # Dynamically plot the new positions. 
            ax.legend(loc='upper left')
        
            # Determine the particle count has a function of the height h.
            count, _ = np.histogram(y, bins=bin_edges)
            
            # Plot the particle count as a function of h.
            ax1.clear()
            ax1.set_ylim(-0.55, 0.55)
            ax1.set_yticks(ticks)
            ax1.set_yticklabels(tick_labels)
            ax1.set_ylabel("height")
            ax1.set_xlabel("counts")
            ax1.barh(h, count, color = 'skyblue', align='center', edgecolor = 'k', height = 0.04)

            # Reset x-axis scaling based on plotted data
            ax1.relim()
            ax1.autoscale_view(scalex=True, scaley=False)
            ax1.annotate(f'Step: {step}', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=12, color='black', 
                         bbox=dict(
                        facecolor='white',    # background color
                        alpha=0.6,            # transparency
                        edgecolor='none')      # no border
                        )
             
            # Code block used for formating to the x tick labels of the bar plot
            # Suppresses the last x tick label so that it doesn't extend past the right 
            # edge of the plot, which can cause jitter in gifs.
            # Get the current x-ticks and convert to string labels
            xticks = ax1.get_xticks()
            xticklabels = [f"{int(tick)}" for tick in xticks]
            # Remove the last label to avoid edge overflow
            if len(xticklabels) > 1:
                xticklabels[-1] = ""
            ax1.set_xticks(xticks)
            ax1.set_xticklabels(xticklabels)
            
            x_bar = np.average(x)
            sigma_x = np.std(x)
            x2_bar = sigma_x**2 + x_bar**2
            y_bar = np.average(y)
            sigma_y = np.std(y)
            y2_bar = sigma_y**2 + y_bar**2
            r2_bar = x2_bar + y2_bar
            r2_history.append(r2_bar)
            ax1.annotate(rf'$\langle x \rangle$ = {x_bar:.5f}', xy=(0.02, -0.4), 
                 xycoords='axes fraction', fontsize=12, color='black')
            ax1.annotate(rf'$\langle x^2 \rangle$ = {x2_bar:.5f}', xy=(0.02, -0.47), 
                 xycoords='axes fraction', fontsize=12, color='black')
            ax1.annotate(rf'$\sigma_x$ = {sigma_x:.5f}', xy=(0.02, -0.54), 
                 xycoords='axes fraction', fontsize=12, color='black')
            ax1.annotate(rf'$\langle y \rangle$ = {y_bar:.5f}', xy=(0.02, -0.65), 
                 xycoords='axes fraction', fontsize=12, color='black')
            ax1.annotate(rf'$\langle y^2 \rangle$ = {y2_bar:.5f}', xy=(0.02, -0.72), 
                 xycoords='axes fraction', fontsize=12, color='black')
            ax1.annotate(rf'$\sigma_y$ = {sigma_y:.5f}', xy=(0.02, -0.79), 
                 xycoords='axes fraction', fontsize=12, color='black')
            ax1.annotate(rf'$\langle r^2 \rangle$ = {r2_bar:.5f}', xy=(0.02, -0.9), 
                 xycoords='axes fraction', fontsize=12, color='black')
                 
            # Optional lines for saving frames of the main figure for assembling a gif or video
            # Comment out the two lines below if the individual frames are not needed
            #os.makedirs("frames", exist_ok=True)
            #fig_main.savefig(f"frames/frame_{step:04d}.png", dpi=600, bbox_inches='tight')
            
        fig_r2 = plt.figure(figsize = (4, 4))
        plt.plot(r2_history, color='purple')
        plt.xlabel("Step")
        plt.ylabel(rf"⟨r²⟩, $\sigma_\mu = $ {mu_std:.4f}")
        plt.grid(True)
        
        fig_nn = plt.figure(figsize = (4, 4))
        plt.plot(nn_history, label='Mean NN Distance', color='green')
        plt.xlabel("Step")
        plt.ylabel(rf"Mean Nearest Neighbor Distance, $\sigma_\mu = $ {mu_std:.4f}")
        plt.grid(True)

        fig_corr = plt.figure(figsize = (4, 4))
        plt.plot(corr_history, label="Clustering Correlation", color = 'pink')
        plt.xlabel("Step")
        plt.ylabel(rf"Divergence × $(C - \langle C\rangle)$, $\sigma_\mu = $ {mu_std:.4f}")
        plt.grid(True)

    return r2_history, nn_history, corr_history, fig_main, ax, ax1, ax2, fig_r2, fig_nn, fig_corr

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run clustering simulation")
    parser.add_argument("--N", type=int, default=10000, help="Number of particles")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of simulation steps")
    parser.add_argument("--mu_std", type=float, default=0.01, help="Std dev of drift field")
    parser.add_argument("--total_std", type=float, default=0.02, help="Total std dev (drift+diffusion)")
    parser.add_argument("--M", type=int, default=50, help="Grid cells along one axis")

    args = parser.parse_args()
    N = args.N
    num_steps = args.num_steps
    mu_std = args.mu_std
    total_std = args.total_std
    M = args.M

    os.makedirs("results_paper", exist_ok=True)
    prefix = f"results_paper/mu_{mu_std:.5e}".replace(".", "_").replace("-", "m")

    with open(f"{prefix}_parameters.txt", "w") as f:
        f.write(f"N = {N}\n")
        f.write(f"num_steps = {num_steps}\n")
        f.write(f"mu_std = {mu_std:.5e}\n")
        f.write(f"total_std = {total_std:.5e}\n")
        f.write(f"M = {M}\n")

    df = pd.DataFrame(columns=["mu_std", "steps", "r2_final", "nn_final", "corr_final"])
    steps = np.linspace(0, num_steps - 1, num_steps)

    # Call the main function
    r2, nn, corr, fig_main, ax, ax1, ax2, fig_r2, fig_nn, fig_corr = simulate_clustering(N = N, num_steps = num_steps, mu_std = mu_std, total_std = total_std, M = M)

    # Use just the last 10% of the data to calculate the Anderson localization metrics.
    # Ensures that stability is reached before characterizing the system.
    tail_fraction = 0.1
    n_tail = int(len(r2) * tail_fraction)

    # Calculate the Anderson localization metrics
    r2_avg = np.mean(r2[-n_tail:])
    nn_avg = np.mean(nn[-n_tail:])
    corr_avg = np.mean(corr[-n_tail:])
    df.loc[len(df)] = [mu_std, num_steps, r2_avg, nn_avg, corr_avg]

    # Save the figures
    fig_main.savefig(f"{prefix}_main.png", dpi=600, bbox_inches='tight')
    fig_r2.savefig(f"{prefix}_r2.png", dpi=600, bbox_inches='tight')
    fig_nn.savefig(f"{prefix}_nn.png", dpi=600, bbox_inches='tight')
    fig_corr.savefig(f"{prefix}_corr.png", dpi=600, bbox_inches='tight')
    # Close the saved figures to prevent memory leaks
    plt.close(fig_main)
    plt.close(fig_r2)
    plt.close(fig_nn)
    plt.close(fig_corr)

    # Store the metric data
    Mr2 = np.transpose(np.matrix([steps, r2]))
    Mnn = np.transpose(np.matrix([steps, nn]))
    Mcorr = np.transpose(np.matrix([steps, corr]))

    # Save the metric data
    np.savetxt(f"{prefix}_r2.txt", Mr2, fmt = "%.6e")
    np.savetxt(f"{prefix}_nn.txt", Mnn, fmt = "%.6e")
    np.savetxt(f"{prefix}_corr.txt", Mcorr, fmt = "%.6e")
    df.to_csv(f"{prefix}_summary.csv", index = False)
    

    # Optional code for saving the individual components of the main figure
    # in a vector graphics format
    #     #for name, axis in zip(["ax", "ax1", "ax2"], [ax, ax1, ax2]):
    #    extent = axis.get_tightbbox(fig_main.canvas.get_renderer())
        # Save each axis as its own vector graphic
    #    fig_main.savefig(f"{prefix}_{name}.pdf", bbox_inches=extent.transformed(fig_main.dpi_scale_trans.inverted()))