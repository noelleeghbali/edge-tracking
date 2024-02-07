from behavior_analysis import trajectory_plotter

# Plot a trajectory
folder_path = '/Users/noelleeghbali/Desktop/exp/tethered_behavior/fall_2023_exp/MAIN/light_crisscross_oct'
select_file = '01182024-125550_PAM_Chr_light_crisscross_oct_Fly31.log'

trajectory_plotter(folder_path, 50, 1050, [-500, 500], [0, 1000], hlines=[350], select_file=select_file, plot_type='odor', save=True)