from behavior_analysis import trajectory_plotter

# Plot a trajectory
folder_path = '/Users/noelleeghbali/Desktop/exp/tethered_behavior/fall_2023_exp/MAIN/light_crisscross_oct'
select_file = None

trajectory_plotter(folder_path, 50, 1050, [-500, 500], [0, 1050], hlines=[350], select_file=None, plot_type='odor', save=True)