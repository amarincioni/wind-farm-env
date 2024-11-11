import floris.tools as ft

import pyglet
from pyglet.sprite import Sprite

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_agg import FigureCanvasAgg


class WindMap(Sprite):
    """
    WindMap is a visualization of an overhead view of the windfarm. It renders the wind farm using matplotlib
    and creates a pyglet Sprite to render from it.
    """

    def __init__(self, size, dpi, cut_plane, turbines_raw_data, wind_speed_limits=None, color_map='GnBu_r', wind_direction=None, flow_points=None, observation_points=None, bounds=None, windfarm_info=None):
        self.fig = plt.figure(figsize=size, dpi=dpi)
        self.fig.tight_layout()
        self.fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        self.ax = self.fig.gca()
        self.ax.set_axis_off()
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)
        self.canvas = FigureCanvasAgg(self.fig)
        self.color_map = color_map
        self.bounds = bounds
        self.windfarm_info = windfarm_info
        if wind_speed_limits is None:
            self.min_wind_speed = self.max_wind_speed = None
        elif hasattr(wind_speed_limits, '__len__') and len(wind_speed_limits) > 1:
            self.min_wind_speed = wind_speed_limits[0]
            self.max_wind_speed = wind_speed_limits[1]
        else:
            raise NotImplementedError
        
        # Power history for plots
        self.power_history_max_length = 20
        self.power_history_padding = 30
        self.max_power = 3.0 # MW
        self.power_plt_height = 300
        self.power_plt_y_offset = 350
        self.power_history = {i: [] for i in range(len(turbines_raw_data))}

        img = self.find_image(cut_plane, turbines_raw_data, wind_direction, flow_points, observation_points)
        super().__init__(img)

    def find_image(self, cut_plane, turbines_raw_data=None, wind_direction=None, flow_points=None, observation_points=None):
        self.clear()
        ft.visualization.visualize_cut_plane(cut_plane, self.ax, self.min_wind_speed, self.max_wind_speed,
                                             self.color_map)
        lines = []
        for i, (angle, coordinates, radius, power) in enumerate(turbines_raw_data):
            line = np.array([[0, -radius], [0, radius]])
            small_line = np.array([[radius / 5, -radius / 5], [radius / 5, radius / 5]])
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.array([[c, s], [-s, c]])
            line = line @ rotation
            small_line = small_line @ rotation
            shift = np.array([[coordinates.x1, coordinates.x2], [coordinates.x1, coordinates.x2]])
            line = line + shift
            small_line = small_line + shift
            lines.append(line)
            lines.append(small_line)
            # Tag each turbine with its id
            self.ax.text(coordinates.x1-70, coordinates.x2-50, f'T{i}', fontsize=8, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.1'))
            
            turbine_info_position = (int(self.bounds[0][1]/2 + 20), self.bounds[1][1])
            # Show list of turbines with their power
            self.ax.text(turbine_info_position[0], turbine_info_position[1]-35*(i+1), f'T{i}: {power/1e6:.2f}MW', fontsize=8, color='black')
            # Add bar for power
            self.ax.plot([int(self.bounds[0][1]/2)+200, int(self.bounds[0][1]/2)+200 + power/1e6*40], [10 + self.bounds[1][1] - 35*(i+1)]*2, 'black', lw=4)
            
            # Add power to logs
            self.power_history[i].append(power/1e6)
            self.power_history[i] = self.power_history[i][-self.power_history_max_length:]

        # Plot power history of all turbines
        # Plot rectangle filled
        self.ax.fill_between([
            int(self.bounds[0][1]/2 + self.power_history_padding), 
            int(self.bounds[0][1] - self.power_history_padding)], 
            [self.bounds[1][1] - self.power_plt_y_offset]*2, [self.bounds[1][1] - self.power_plt_y_offset - self.power_plt_height]*2, color='gray', alpha=0.5)
        # Plot y axis
        self.ax.plot([int(self.bounds[0][1]/2 + self.power_history_padding)]*2, [self.bounds[1][1] - self.power_plt_y_offset - self.power_plt_height, self.bounds[1][1] - self.power_plt_y_offset], 'black', lw=2)
        # Plot y axis values
        self.ax.text(int(self.bounds[0][1]/2 + self.power_history_padding), self.bounds[1][1] - self.power_plt_y_offset - self.power_plt_height - 20, '0MW', fontsize=8, color='black')
        self.ax.text(int(self.bounds[0][1]/2 + self.power_history_padding), self.bounds[1][1] - self.power_plt_y_offset + 10, f'{self.max_power}MW', fontsize=8, color='black')
        
        for i, power_history in self.power_history.items():
            box_len = int(self.bounds[0][1]/2 - 2*self.power_history_padding)
            x = np.arange(len(power_history))*int(box_len/self.power_history_max_length) + int(self.bounds[0][1]/2+self.power_history_padding)
            y = np.array(power_history)/self.max_power*self.power_plt_height + self.bounds[1][1] - self.power_plt_y_offset - self.power_plt_height
            self.ax.plot(x, y, label=f'Turbine {i}')
        self.ax.legend(loc='upper right') 

        lc = LineCollection(lines, color="black", lw=4)
        self.ax.add_collection(lc)

        # Add windfarm bounds (use windfarm info bounds)
        if observation_points is not None:
            # Dashed windfarm bounds
            self.ax.plot([self.windfarm_info["bounds"][0][0], self.windfarm_info["bounds"][1][0]], [self.windfarm_info["bounds"][0][1], self.windfarm_info["bounds"][0][1]], 'black', lw=2, linestyle='dashed', alpha=0.2)
            self.ax.plot([self.windfarm_info["bounds"][0][0], self.windfarm_info["bounds"][1][0]], [self.windfarm_info["bounds"][1][1], self.windfarm_info["bounds"][1][1]], 'black', lw=2, linestyle='dashed', alpha=0.2)
            self.ax.plot([self.windfarm_info["bounds"][0][0], self.windfarm_info["bounds"][0][0]], [self.windfarm_info["bounds"][0][1], self.windfarm_info["bounds"][1][1]], 'black', lw=2, linestyle='dashed', alpha=0.2)
            self.ax.plot([self.windfarm_info["bounds"][1][0], self.windfarm_info["bounds"][1][0]], [self.windfarm_info["bounds"][0][1], self.windfarm_info["bounds"][1][1]], 'black', lw=2, linestyle='dashed', alpha=0.2)
            # Plot with the margin
            self.ax.plot([self.windfarm_info["bounds"][0][0] - self.windfarm_info["margin"], self.windfarm_info["bounds"][1][0] + self.windfarm_info["margin"]], [self.windfarm_info["bounds"][0][1] - self.windfarm_info["margin"], self.windfarm_info["bounds"][0][1] - self.windfarm_info["margin"]], 'black', lw=2, alpha=0.2)
            self.ax.plot([self.windfarm_info["bounds"][0][0] - self.windfarm_info["margin"], self.windfarm_info["bounds"][1][0] + self.windfarm_info["margin"]], [self.windfarm_info["bounds"][1][1] + self.windfarm_info["margin"], self.windfarm_info["bounds"][1][1] + self.windfarm_info["margin"]], 'black', lw=2, alpha=0.2)
            self.ax.plot([self.windfarm_info["bounds"][0][0] - self.windfarm_info["margin"], self.windfarm_info["bounds"][0][0] - self.windfarm_info["margin"]], [self.windfarm_info["bounds"][0][1] - self.windfarm_info["margin"], self.windfarm_info["bounds"][1][1] + self.windfarm_info["margin"]], 'black', lw=2, alpha=0.2)
            self.ax.plot([self.windfarm_info["bounds"][1][0] + self.windfarm_info["margin"], self.windfarm_info["bounds"][1][0] + self.windfarm_info["margin"]], [self.windfarm_info["bounds"][0][1] - self.windfarm_info["margin"], self.windfarm_info["bounds"][1][1] + self.windfarm_info["margin"]], 'black', lw=2, alpha=0.2)

        # Add observation points
        if observation_points is not None:
            x, y = observation_points["x"], observation_points["y"]
            u = observation_points["u"]
            self.ax.scatter(x, y, c=u, cmap='viridis', s=60)

            # Plot current timestep
            t = observation_points["t"].max()
            self.ax.text(self.bounds[0][0] + 10, self.bounds[1][1] - 10, f'Timestep: {t}', fontsize=8, color='black')

            # for each turbine, plot only the observation points behind it
            # circle them according to the turbine id
            # if theyre close enoguh to the turbine
            # for i, (angle, coordinates, radius, power) in enumerate(turbines_raw_data):
            #     # get the points behind the turbine
            #     x = np.copy(observation_points["x"])
            #     y = np.copy(observation_points["y"])
            #     behind = np.where(np.cos(angle) * (x - coordinates.x1) + np.sin(angle) * (y - coordinates.x2) < 0)
            #     close = np.where(np.sqrt((x - coordinates.x1)**2 + (y - coordinates.x2)**2) < 1.5*radius)
            #     close = close[0]
            #     behind = behind[0]  
            #     plot_ids = [i for i in close if i in behind]
            #     x_plot = x[plot_ids]
            #     y_plot = y[plot_ids]

            #     # plot them
            #     colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'pink', 'brown', 'gray', 'black']
            #     self.ax.scatter(x_plot, y_plot, c=colors[i%len(colors)], s=65)                    
            #     circle = plt.Circle((coordinates.x1, coordinates.x2), int(1.5*radius), color=colors[i%len(colors)], fill=False)
            #     self.ax.add_artist(circle)

        # Add wind direction line
        if wind_direction is not None:
            wind_direction = -wind_direction
            wind_direction_line = np.array([[0, -30], [0, 40]])
            c, s = np.cos(np.deg2rad(wind_direction)), np.sin(np.deg2rad(wind_direction))
            rotation = np.array([[c, s], [-s, c]])
            wind_direction_line = wind_direction_line @ rotation
            middle_x = (self.windfarm_info["bounds"][1][0] - self.windfarm_info["bounds"][0][0])/2
            position = (middle_x, self.bounds[1][1] - 50)
            wind_direction_line = wind_direction_line + position
            # Draw circle around wind direction
            circle = plt.Circle(position, 40, color=(0.3, 0, 0), fill=False, 
                                lw=2, alpha=0.5)
            self.ax.add_artist(circle)
            self.ax.arrow(wind_direction_line[1, 0], wind_direction_line[1, 1],
                          wind_direction_line[0, 0] - wind_direction_line[1, 0],
                          wind_direction_line[0, 1] - wind_direction_line[1, 1],
                          head_width=10, head_length=10, fc='red', ec='red', lw=2)
            self.ax.text(position[0] - 90, position[1] - 65, 'Wind direction', fontsize=8, color='black')

        # Add dots at the flow points
        if flow_points is not None:
            for point in np.array(flow_points).T:
                self.ax.plot(point[0], point[1], 'rx', markersize=3)

        raw_data, size = self.canvas.print_to_buffer()
        width = size[0]
        height = size[1]
        return pyglet.image.ImageData(width, height, 'RGBA', raw_data, -4 * width)

    def update_image(self, cut_plane, turbines_raw_data=None, wind_direction=None, flow_points=None, observation_points=None):
        self.image = self.find_image(cut_plane, turbines_raw_data, wind_direction, flow_points, observation_points)

    def render(self):
        self.draw()

    def clear(self):
        self.ax.clear()

    def close(self):
        plt.close(self.fig)
