import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


def open_pickle(filename):  # Open a pickle of preprocessed imaging data for a cohort of flies
    return pd.read_pickle(filename)
    # This will return a dictionary with keys for each fly that was imaged; the corresponding values themselves are
    # dictionaries with keys 'a1', 'd', 'di', 'do', 'ft', all with values storing dataframes


def plot_trial_FF_odor(figure_folder, filename, lobes, colors, keyword=None):
    # Load data
    data = open_pickle(filename)
    flylist_n = np.array(list(data.keys()))

    # Determine flies to plot
    if keyword is not None and 1 <= keyword <= len(flylist_n):
        # Plotting for a specific fly
        flies_to_plot = [(keyword, data[flylist_n[keyword - 1]])]
    else:
        # Plotting for all flies
        flies_to_plot = [(fly_key, this_fly) for fly_key, this_fly in enumerate(data.values(), start=1)]

    # Plot data
    for fly_number, this_fly in flies_to_plot:
        # Assign the correct dataframe ('a1' is all the data collected)
        this_fly_all_data = this_fly['a1']

        # Create a figure
        fig, axs = plt.subplots(len(lobes), 1, figsize=(10, 2 * len(lobes)), sharex=True)

        # Set the font to Arial for all text
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Arial'

        # Iterate through the lobes
        for i, (lobe, color) in enumerate(zip(lobes, colors)):
            # Assign FF (fluorescence) to the correct df column
            FF = this_fly_all_data[lobe]
            # Extract time
            time = this_fly_all_data['relative_time']
            # Plot FF over time
            axs[i].plot(time, FF, color=color, linewidth=1)
            axs[i].set_ylabel(f'{lobe} dF/F')
            axs[i].grid(False)

            # Assign the correct dataframe ('di' is all the in odor data')
            d_odor = this_fly['di']
            for key, df in d_odor.items():
                # Find the first and last timepoints of every bout
                time_on = df['relative_time'].iloc[0]
                time_off = df['relative_time'].iloc[-1]
                # Block that time out
                timestamp = time_off - time_on
                # Add a rectangle highlight of the length of the odor stimulus to denote when odor is on
                rectangle = patches.Rectangle((time_on, FF.min()), timestamp, FF.max() + 0.5, facecolor='#ff7f24', alpha=0.3)
                axs[i].add_patch(rectangle)

        # Set title with fly number
        title = f'fly {fly_number}'
        plt.suptitle(f'{title} trial dF/F')
        plt.xlabel('time (sec)')
        plt.savefig(os.path.join(figure_folder, f'{title}_FF_odor'))
        plt.clf()


def plot_triggered_norm_FF(figure_folder, filename, lobes, colors, window_size=5, event_type='entry'):
    data = open_pickle(filename)
    flylist_n = np.array(list(data.keys()))

    # Create a figure
    fig, axs = plt.subplots(1, len(lobes), figsize=(4 * len(lobes), 5), sharex=True, sharey=True)

    # Set the font to Arial for all text
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'

    # Iterate through the lobes, assigning variables like in the previous function
    for i, (lobe, color) in enumerate(zip(lobes, colors)):
        for fly_n, fly_key in enumerate(flylist_n, start=1):
            this_fly = data[fly_key]
            this_fly_all_data = this_fly['a1']

            FF = this_fly_all_data[lobe]
            time = this_fly_all_data['relative_time']

            # Choose the dataset based on event type
            if event_type == 'entry':
                d_event = this_fly['di']
            else:
                d_event = this_fly['do']

            for key, df in d_event.items():
                time_on = df['relative_time'].iloc[0]

                # Extract a window around the event
                window_start = time_on - window_size / 2
                window_end = time_on + window_size / 2
                window_mask = (time >= window_start) & (time <= window_end)

                # Check if data points fall within the window
                if any(window_mask):
                    # Z-score fluorecence
                    normalized_FF = stats.zscore(FF[window_mask])
                    time_aligned = time[window_mask] - time_on
                    # Plot fluorescence aligned to the event time
                    axs[i].plot(time_aligned, normalized_FF, color=color, alpha=0.1, linewidth=0.2)
        
        # Customize the plot
        axs[i].set_title(lobe)
        axs[i].set_ylim(-3, 8)
        axs[i].set_ylabel('dF/F')
        axs[i].set_xlabel('time (sec)')
        axs[i].grid(False)
        axs[i].vlines(x=0, ymin=-5, ymax=10, color='grey', alpha=0.5, linestyles='--')

    plt.suptitle(f'normalized dF/F at {event_type}')
    plt.savefig(os.path.join(figure_folder, f'norm_FF_{event_type}'))
