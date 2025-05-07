import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
import pickle

class Visualiser:

    def __init__(self, detect_start=None, detect_end=None):
        """
        Class that visualises the traffic flow evolution.
        :param traffic_evolution: TxN matrix of traffic evolution
        """
        self.detect_start = detect_start
        self.detect_end = detect_end
        plt.rcParams.update({
            "axes.titlesize": 18,
            "axes.labelsize": 12,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        })

    @staticmethod
    def read_csv_grid(filename, dtype=float):
        """
        Reads a 2D grid from a CSV file into a NumPy array.

        Args:
            filename (str): The base name of the input file ('.csv' will be added).
            dtype (type): The desired data type for the NumPy array elements
                          (e.g., float, int). Defaults to float.

        Returns:
            np.ndarray: A 2D NumPy array containing the data, or None if error.
        """
        full_path = f"{filename}.csv"
        print(f"Reading 2D data from {full_path}...")
        data = []

        with open(full_path, "r") as f:
            reader = csv.reader(f)
            for row_str in reader:
                processed_row = [dtype(item) for item in row_str]
                data.append(processed_row)
        print("Read successful.")
        return np.array(data, dtype=dtype)

    @staticmethod
    def read_pickle(filename):
        with open(filename, 'rb') as f:  # Open in binary read mode
            loaded_data = pickle.load(f)
        return loaded_data

    def create_gif(self, traffic_evolution, light_positions=None, light_state_history=None):
        plt.rcParams.update({"xtick.labelsize": 8})
        fig, axis = plt.subplots()
        # Set x- and y-axis
        axis.set_xlim(0, traffic_evolution.shape[1] - 0.5)
        axis.set_ylim(0, 1)

        # Sets labels and ticks on the x-axis and hides y-axis labels
        axis.set_xticks(np.arange(traffic_evolution.shape[1]))
        axis.set_xticks(np.arange(traffic_evolution.shape[1]) - 0.5, minor=True)
        axis.set_yticks([])

        axis.set_aspect("equal")

        # Shows grid for the final animation
        axis.grid(True, which="minor", color="black", linestyle='-', linewidth=1)

        # initial plot for the animation is the initial state of the system
        animated_plot = axis.imshow(
            [traffic_evolution[0, :]], cmap="gray_r", vmin=0, vmax=2, aspect="equal",
            extent=[-0.5, traffic_evolution.shape[1] - 0.5, 0, traffic_evolution.shape[0]], zorder=0
        )

        # updates the data which is plotted for every frame (frame corresponds to the time step in the model)
        def update_data(frame):
            if light_positions is not None:
                states_in_frame = light_state_history[frame]
                for light_pos in light_positions:
                    axis.axvspan(light_pos - 0.5, light_pos + 0.5,
                                 color="white")
                    light_is_green = states_in_frame.get(light_pos, False)

                    color = "green" if light_is_green else "red"
                    axis.axvspan(light_pos-0.5, light_pos+0.5,
                                     color=color, alpha=0.3)
            if frame > 0:
                animated_plot.set_data([traffic_evolution[frame, :]])
            return animated_plot,

        animation = FuncAnimation(
            fig=fig,
            func=update_data,
            frames=traffic_evolution.shape[0],
            interval=1000,
            repeat=False
        )

        animation.save("CellAutomata/traffic_visualisation.gif")

    def matrix_plot(self, traffic_evolution, light_positions=None, light_state_history=None):
        fig, axis = plt.subplots()

        # plots the matrix values (white=empty, gray=car, gridlines=black)
        axis.imshow(traffic_evolution, cmap="gray_r", vmin=0, vmax=2, aspect="auto")

        # sets labels on x- and y-axis
        #axis.set_xticks(np.arange(-0.5, traffic_evolution.shape[1], 1), minor=True)
        #axis.set_yticks(np.arange(-0.5, traffic_evolution.shape[0], 1), minor=True)

        # Enable full grid
        #axis.grid(which="minor", color="black", linestyle='-', linewidth=1)

        # Align ticks with cell edges
        #axis.set_xticks(np.arange(traffic_evolution.shape[1]))
        #axis.set_yticks(np.arange(0, traffic_evolution.shape[0], 2))

        if light_positions is not None:
            for i in range(len(light_positions)):
                light_pos = light_positions[i]
                for time_step in range(traffic_evolution.shape[0]):
                    states_in_frame = light_state_history[time_step]
                    light_is_green = states_in_frame.get(light_pos, False)

                    color = "green" if light_is_green else "red"
                    axis.axvspan(
                        light_pos - 0.5, light_pos + 0.5,
                        1 - ((time_step + 1) * (1 / traffic_evolution.shape[0])),
                        1 - (time_step * (1 / traffic_evolution.shape[0])),
                        color=color, alpha=0.3
                    )
        axis.set_xlabel("Road Position")
        axis.set_ylabel("Time Step")

        plt.show()

    def density_meanvel_flow_plot(self, filename, fixed_value):
        """
        Plots the general density vs mean velocity / flow graph.
        :param fixed_value: String of the form vmax=... or p=...
        """
        results = self.read_pickle(filename)
        fig, (ax_f, ax_v) = plt.subplots(2, 1, figsize=(8, 10), sharex=False)
        for label, data in results.items():
            if fixed_value in label:
                parts = label.split(',')
                part1 = parts[0].strip()
                part2 = parts[1].strip()
                if fixed_value in part1:
                    plot_label = part2
                elif fixed_value in part2:
                    plot_label = part1
                ax_f.plot(data['densities'], data['flows'], marker='.', linestyle='-', label=plot_label)
                ax_v.plot(data['densities'], data['velocities'], marker='.', linestyle='-', label=plot_label)

        ax_f.set_ylabel("Average Flow [vehicles/timestep]")
        ax_f.set_xlabel("Density [vehicles/cell]")
        ax_f.legend()

        ax_v.set_ylabel("Average Velocity [cells/timestep]")
        ax_v.set_xlabel("Density [vehicles/cell]")
        ax_v.legend()
        plt.show()

    def traffic_light_cycle_flow_sync_plot(self, num_cars_list, road_length, cycle_lengths, flow_list):
        plt.figure()
        for k in range(len(flow_list)):
            plt.plot(cycle_lengths, flow_list[k], label=f"density={num_cars_list[k]/road_length}")
        plt.xlabel("Cycle Length")
        plt.ylabel("Flow [vehicles/timestep]")
        plt.legend(loc="upper right")
        plt.show()

    def red_green_proportion_plot(self, results_path):
        results = self.read_pickle(results_path)
        plt.figure()
        for T, data in results.items():
            green_durations = np.array(data['green_durations'])
            actual_fractions = green_durations / T
            plt.plot(actual_fractions, data['flows'], marker='.', markersize=5,
                     linestyle='-', label=f'Total Cycle = {T}')

        plt.xlabel("Green Time Fraction")
        plt.ylabel("Average Flow [vehicles/timestep]")
        plt.legend()
        plt.ylim(bottom=0)
        plt.xlim(0, 1)
        plt.show()

    def traffic_light_delay_flow_plot(self, num_cars_list, road_length, offset_range, flow_list):
        plt.figure()
        for k in range(len(flow_list)):
            plt.plot(offset_range, flow_list[k], label=f"density={num_cars_list[k]/road_length}")
        plt.xlabel("Time delay")
        plt.ylabel("Flow [vehicles/timestep]")
        #plt.legend()
        plt.show()

    def traffic_light_cycle_flow_delay(self, cycle_lengths, offsets, flow_list, velocity_list):
        plt.figure()
        for k in range(len(flow_list)):
            plt.plot(cycle_lengths, flow_list[k], label=f"Time delay: {offsets[k]}")
        plt.xlabel("Cycle Length")
        plt.ylabel("Flow [vehicles/timestep]")
        plt.legend()
        plt.show()

        plt.figure()
        for k in range(len(flow_list)):
            plt.plot(cycle_lengths, velocity_list[k], label=f"Time delay: {offsets[k]}")
        plt.xlabel("Cycle Length")
        plt.ylabel("Velocity [cells/timestep]")
        plt.legend()
        plt.show()


    def flow_braking_prob(self, p_values, avg_flow):
        plt.figure()
        plt.plot(p_values, avg_flow)
        plt.xlabel("Braking probability")
        plt.ylabel("Average flow [vehicles/timestep]")
        plt.show()

    def sotl_parameter_influence_onefixed(self, parameter_to_vary, parameter_values, flow):
        plt.figure()
        plt.plot(parameter_values, flow)
        plt.xlabel(parameter_to_vary)
        plt.ylabel("Flow [vehicles/timestep]")
        plt.show()

    def sotl_parameter_influence_grid(self, threshold_values, distance_values, path_to_grid):
        X, Y = np.meshgrid(distance_values, threshold_values)
        fig, ax = plt.subplots(figsize=(8, 6))
        flows_grid = self.read_csv_grid(path_to_grid)

        contour = ax.contourf(X, Y, flows_grid, cmap='viridis', levels=20)

        cbar = fig.colorbar(contour)
        cbar.set_label('Average Flow [vehicles/timestep]')

        # Label the axes
        ax.set_xlabel("Detection Distance (d) [cells]")
        ax.set_ylabel("Queue Threshold (N_wait)")
        plt.show()

    def compare_sync_gw(self, results):
        plt.figure()
        plt.plot(results["densities"], results["gw_flows"], label="Green wave", color="green")
        plt.plot(results["densities"], results["sync_flows"], label="Synchronised")
        plt.xlabel("Density [vehicles/cell]")
        plt.ylabel("Flow [vehicles/timestep]")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(results["densities"], results["gw_velocities"], label="Green wave", color="green")
        plt.plot(results["densities"], results["sync_velocities"], label="Synchronised")
        plt.xlabel("Density [vehicles/cell]")
        plt.ylabel("Velocity [cells/timestep]")
        plt.legend()
        plt.show()

    def compare_gw_sotl(self, results):
        plt.figure()
        plt.plot(results["densities"], results["gw_flows"], label="Green wave", color="green")
        plt.plot(results["densities"], results["sotl_flows"], label="Self organised")
        plt.xlabel("Density [vehicles/cell]")
        plt.ylabel("Flow [vehicles/timestep]")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(results["densities"], results["gw_velocities"], label="Green wave", color="green")
        plt.plot(results["densities"], results["sotl_velocities"], label="Self organised")
        plt.xlabel("Density [vehicles/cell]")
        plt.ylabel("Velocity [cells/timestep]")
        plt.legend()
        plt.show()