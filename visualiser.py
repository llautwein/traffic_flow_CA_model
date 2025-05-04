import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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
            "xtick.labelsize": 12,
            "ytick.labelsize": 8,
            "legend.fontsize": 12,
        })

    def create_gif(self, traffic_evolution, light_positions=None, light_state_history=None):
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

    def density_meanvel_flow_plot(self, density, mean_velocity, variance_velocity, flow):
        plt.figure()
        plt.plot(density, mean_velocity)
        plt.xlabel("Density [vehicles/cell]")
        plt.ylabel("Mean velocity [cell/timestep]")
        plt.show()

        plt.figure()
        plt.plot(density, flow)
        plt.xlabel("Density [vehicles/cell]")
        plt.ylabel("Flow [vehicles/timestep]")
        plt.show()

        plt.figure()
        plt.plot(density, variance_velocity)
        plt.xlabel("Density [vehicles/cell]")
        plt.ylabel("Velocity variance")
        plt.show()

    def traffic_light_cycle_flow_sync_plot(self, num_cars_list, road_length, cycle_lengths, flow_list):
        plt.figure()
        for k in range(len(flow_list)):
            plt.plot(cycle_lengths, flow_list[k], label=f"density={num_cars_list[k]/road_length}")
        plt.xlabel("Cycle Length")
        plt.ylabel("Flow [vehicles/timestep]")
        plt.legend()
        plt.show()

    def traffic_light_delay_flow_plot(self, num_cars_list, road_length, offset_range, flow_list):
        plt.figure()
        for k in range(len(flow_list)):
            plt.plot(offset_range, flow_list[k], label=f"density={num_cars_list[k]/road_length}")
        plt.xlabel("Time delay")
        plt.ylabel("Flow [vehicles/timestep]")
        plt.legend()
        plt.show()

    def traffic_light_cycle_flow_delay(self, cycle_lengths, offsets, flow_list):
        plt.figure()
        for k in range(len(flow_list)):
            plt.plot(cycle_lengths, flow_list[k], label=f"Time delay: {offsets[k]}")
        plt.xlabel("Cycle Length")
        plt.ylabel("Flow [vehicles/timestep]")
        plt.legend()
        plt.show()


    def flow_braking_prob(self, p_values, avg_flow):
        plt.figure()
        plt.plot(p_values, avg_flow)
        plt.xlabel("Braking probability")
        plt.ylabel("Average flow [vehicles/timestep]")
        plt.show()
