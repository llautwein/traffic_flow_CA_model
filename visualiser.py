import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Visualiser:

    def __init__(self, traffic_evolution, detect_start=None, detect_end=None):
        """
        Class that visualises the traffic flow evolution.
        :param traffic_evolution: TxN matrix of traffic evolution
        """
        self.traffic_evolution = traffic_evolution
        self.detect_start = detect_start
        self.detect_end = detect_end
        plt.rcParams.update({
            "axes.titlesize": 18,
            "axes.labelsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 8,
            "legend.fontsize": 12,
        })

    def create_gif(self):
        fig, axis = plt.subplots()
        # Set x- and y-axis
        axis.set_xlim(0, self.traffic_evolution.shape[1] - 0.5)
        axis.set_ylim(0, 1)

        # Sets labels and ticks on the x-axis and hides y-axis labels
        axis.set_xticks(np.arange(self.traffic_evolution.shape[1]))
        axis.set_xticks(np.arange(self.traffic_evolution.shape[1]) - 0.5, minor=True)
        axis.set_yticks([])

        axis.set_aspect("equal")

        # Shows grid for the final animation
        axis.grid(True, which="minor", color="black", linestyle='-', linewidth=1)

        # Overlay the detector region
        if self.detect_start is not None and self.detect_end is not None:
            axis.axvspan(self.detect_start - 0.5, self.detect_end + 0.5,
                         color='red', alpha=0.3, zorder=1)

        # initial plot for the animation is the initial state of the system
        animated_plot = axis.imshow(
            [self.traffic_evolution[0, :]], cmap="gray_r", vmin=0, vmax=2, aspect="equal",
            extent=[-0.5, self.traffic_evolution.shape[1] - 0.5, 0, self.traffic_evolution.shape[0]], zorder=0
        )

        # updates the data which is plotted for every frame (frame corresponds to the time step in the model)
        def update_data(frame):
            if frame > 0:
                animated_plot.set_data([self.traffic_evolution[frame, :]])
            return animated_plot,

        animation = FuncAnimation(
            fig=fig,
            func=update_data,
            frames=self.traffic_evolution.shape[0],
            interval=1000,
            repeat=False
        )

        animation.save("CellAutomata/traffic_visualisation.gif")

    def matrix_plot(self):
        fig, axis = plt.subplots()

        # plots the matrix values (white=empty, gray=car, gridlines=black)
        axis.imshow(self.traffic_evolution, cmap="gray_r", vmin=0, vmax=2, aspect="auto")

        # sets labels on x- and y-axis
        axis.set_xticks(np.arange(-0.5, self.traffic_evolution.shape[1], 1), minor=True)
        axis.set_yticks(np.arange(-0.5, self.traffic_evolution.shape[0], 1), minor=True)

        # Enable full grid
        axis.grid(which="minor", color="black", linestyle='-', linewidth=1)

        # Align ticks with cell edges
        axis.set_xticks(np.arange(self.traffic_evolution.shape[1]))
        axis.set_yticks(np.arange(0, self.traffic_evolution.shape[0], 2))

        axis.set_xlabel("Road Position")
        axis.set_ylabel("Time Step")

        plt.show()
