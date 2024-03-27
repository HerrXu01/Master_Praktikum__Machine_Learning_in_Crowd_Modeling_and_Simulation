import scipy.spatial.distance
from PIL import Image, ImageTk
import numpy as np
import math
import skfmm
# pip install scikit-fmm

class Pedestrian:
    """
    Defines a single pedestrian.
    """
    def __init__(self, position, desired_speed):
        self._position = position
        self._desired_speed = desired_speed
        self.clock = 0
        self.went_distance = 0
        self.cumulative_distance = 0

        self.completed = False

    @property
    def position(self):
        return self._position

    @property
    def desired_speed(self):
        return self._desired_speed

    def get_neighbors(self, scenario):
        """
        Compute all neighbors in a 9 cell neighborhood of the current position.
        :param scenario: The scenario instance.
        :return: A list of neighbor cell indices (x,y) around the current position.
        """
        return [
            (int(x + self._position[0]), int(y + self._position[1]))
            for x in [-1, 0, 1]
            for y in [-1, 0, 1]
            if 0 <= x + self._position[0] < scenario.width and 0 <= y + self._position[1] < scenario.height and np.abs(x) + np.abs(y) > 0
        ]

    def update_step_task1(self, scenario):
        """
        Moves to the cell with the lowest distance to the target.
        This does not take obstacles or other pedestrians into account.
        Pedestrians can occupy the same cell.

        :param scenario: The current scenario instance.
        """
        neighbors = self.get_neighbors(scenario)
        next_cell_distance = scenario.target_distance_grids[self._position[0]][self._position[1]]
        next_pos = self._position
        for (n_x, n_y) in neighbors:
            if scenario.grid[n_x, n_y] == scenario.NAME2ID["OBSTACLE"]:
                continue
            if next_cell_distance > scenario.target_distance_grids[n_x, n_y]:
                next_pos = (n_x, n_y)
                next_cell_distance = scenario.target_distance_grids[n_x, n_y]
        self._position = next_pos

    def update_step(self, scenario):
        """
        Moves to the cell with the lowest distance to the target.
        This does not take obstacles or other pedestrians into account.
        Pedestrians can occupy the same cell.

        :param scenario: The current scenario instance.
        """
        # neighbors = self.get_neighbors(scenario)
        # next_cell_distance = scenario.target_distance_grids[self._position[0]][self._position[1]]
        # next_pos = self._position
        # for (n_x, n_y) in neighbors:
        #     if next_cell_distance > scenario.target_distance_grids[n_x, n_y]:
        #         next_pos = (n_x, n_y)
        #         next_cell_distance = scenario.target_distance_grids[n_x, n_y]
        # self._position = next_pos

        p_update = False
        usable_distance = self._desired_speed + self.cumulative_distance
        self.cumulative_distance = 0
        while usable_distance:
            neighbors = self.get_neighbors(scenario)
            next_cell_distance = scenario.target_distance_grids[self._position[0], self._position[1]]
            next_pos = self._position
            # Goes to the neighbor position that minimizes the distance to the nearest target
            for n_x, n_y in neighbors:
                if scenario.grid[n_x, n_y] != Scenario.NAME2ID['OBSTACLE'] and scenario.grid[n_x, n_y] != \
                        Scenario.NAME2ID['PEDESTRIAN']:
                    if next_cell_distance == scenario.target_distance_grids[n_x, n_y]:
                        to_neighbor = math.sqrt((self._position[0] - n_x) ** 2 + (self._position[1] - n_y) ** 2)
                        to_next = math.sqrt((self._position[0] - next_pos[0]) ** 2 + (self._position[1] - next_pos[1]) ** 2)
                        if to_neighbor < to_next:
                            next_pos = (n_x, n_y)
                            next_cell_distance = scenario.target_distance_grids[n_x, n_y]
                    elif next_cell_distance > scenario.target_distance_grids[n_x, n_y]:
                        next_pos = (n_x, n_y)
                        next_cell_distance = scenario.target_distance_grids[n_x, n_y]

            if self._position != next_pos:
                # Assuming next_pos is a tuple and self._position is a tuple as well
                x_next, y_next = next_pos
                x_self, y_self = self._position
                matching_points = [(x, y, size) for x, y, size in scenario.measurement_p
                                   if (x < x_next < x + size) and (y < y_next < y + size) and (x_self < x)]
                for point in matching_points:
                    scenario.measurement[point].append(self.clock)

                p_update = True
                to_go = math.sqrt((self._position[0] - next_pos[0]) ** 2 + (self._position[1] - next_pos[1]) ** 2)

                if to_go <= usable_distance:
                    self.went_distance = self.went_distance + to_go
                    usable_distance = usable_distance - to_go
                    if scenario.grid[next_pos] == Scenario.NAME2ID['TARGET']:
                        self.completed = True
                    else:
                        scenario.grid[next_pos] = Scenario.NAME2ID['PEDESTRIAN']
                    scenario.grid[self._position] = Scenario.NAME2ID['EMPTY']
                    self._position = next_pos
                else:
                    self.cumulative_distance = self.cumulative_distance + usable_distance
                    usable_distance = 0

            else:
                break

        self.clock = self.clock + 1
        return p_update
class Scenario:
    """
    A scenario for a cellular automaton.
    """
    GRID_SIZE = (500, 500)
    measurement = {}
    measurement_p = []
    max_distance = 1
    ID2NAME = {
        0: 'EMPTY',
        1: 'TARGET',
        2: 'OBSTACLE',
        3: 'PEDESTRIAN'
    }
    NAME2COLOR = {
        'EMPTY': (255, 255, 255),
        'PEDESTRIAN': (255, 0, 0),
        'TARGET': (0, 0, 255),
        'OBSTACLE': (255, 0, 255)
    }
    NAME2ID = {
        ID2NAME[0]: 0,
        ID2NAME[1]: 1,
        ID2NAME[2]: 2,
        ID2NAME[3]: 3
    }

    def __init__(self, width, height):
        if width < 1 or width > 1024:
            raise ValueError(f"Width {width} must be in [1, 1024].")
        if height < 1 or height > 1024:
            raise ValueError(f"Height {height} must be in [1, 1024].")

        self.width = width
        self.height = height
        self.grid_image = None
        self.grid = np.zeros((width, height))
        self.pedestrians = []
        self.compute_method = 2  # basic update
        self.target_distance_grids = self.recompute_target_distances()


    def recompute_target_distances(self):
        """
        Calculates the target distance using the specified algorithm.
        """
        # print(self.compute_method)
        if self.compute_method == 0:
            # print('DIJKSTRA!!!')
            self.target_distance_grids = self.dijkstra_update_target_grid()
        elif self.compute_method == 1:
            # print('FMM!!!')
            self.target_distance_grids = self.fmm_update_target_grid()
        elif self.compute_method == 2:
            self.target_distance_grids = self.update_target_grid()
            return self.target_distance_grids

    def update_target_grid(self):
        """
        Computes the shortest distance from every grid point to the nearest target cell.
        This does not take obstacles into account.
        :returns: The distance for every grid cell, as a np.ndarray.
        """
        targets = []
        for x in range(self.width):
            for y in range(self.height):
                if self.grid[x, y] == Scenario.NAME2ID['TARGET']:
                    targets.append([y, x])  # y and x are flipped because they are in image space.
        if len(targets) == 0:
            return np.zeros((self.width, self.height))
        
        # print(targets)

        targets = np.row_stack(targets)

        # print(targets)
        # print(targets.shape)

        x_space = np.arange(0, self.width)
        y_space = np.arange(0, self.height)
        xx, yy = np.meshgrid(x_space, y_space)
        positions = np.column_stack([xx.ravel(), yy.ravel()])

        #positions is finally a (2d array) of all the coordinates possible for a given scenario.

        # print(x_space)
        # print(y_space)
        # print(xx)
        # print(yy)
        # print(positions)

        # after the target positions and all grid cell positions are stored,
        # compute the pair-wise distances in one step with scipy.
        distances = scipy.spatial.distance.cdist(targets, positions)

        # print(distances.shape)

        # now, compute the minimum over all distances to all targets.
        distances = np.min(distances, axis=0)
        # print(distances.shape)

        return distances.reshape((self.width, self.height))

    """         DIJKSTRA         """
    def minimum_distance(self, distances, position):
        """
        Calculate the minimum distance to the current location considering all neighbors in the 9-cell neighborhood of the current location on the distance map.
        :param distances:2D nparray distance table
        :param position:A tuple position (x,y)
        :return:A list of neighbor cell indices (x,y) around the current position.
        """
        new_distance = distances[position[0], position[1]]
        for x_offset in [-1, 0, 1]:
            for y_offset in [-1, 0, 1]:
                x, y = position[0] + x_offset, position[1] + y_offset  # [position-1,position,position+1]
                if 0 <= x < self.width and 0 <= y < self.height:
                    temp_distance = distances[x, y] + (1 if x == position[0] or y == position[1] else math.sqrt(2))
                    new_distance = min(new_distance, temp_distance)
        return new_distance
    def accessible_neighbors(self, closed, position):
        """
        Counts all neighbors in the 9-cell neighborhood of the current location.
        :param closed:2D binary table marking closing positions
        :param position:A tuple position (x,y)
        :return:A list of neighbor cell indices (x,y) around the current position.
        """
        neighbors = [(x, y) for x in range(position[0] - 1, position[0] + 2)
                     for y in range(position[1] - 1, position[1] + 2)
                     if (0 <= x < self.width) and (0 <= y < self.height) and closed[x, y] == 0]
        for x, y in neighbors:
            closed[x, y] = 1
        return neighbors

    def dijkstra_update_target_grid(self):
        """
        Calculate the shortest distance from each grid point to the nearest target cell.
        This does not take barriers into account.
        :return: the distance of each grid cell.
        """
        # print('dijkstra')
        register = 1
        targets = [(x, y) for x in range(self.width) for y in range(self.height) if
                   self.grid[x, y] == Scenario.NAME2ID['TARGET']]
        closed = np.where(self.grid == Scenario.NAME2ID['OBSTACLE'], 1, 0)

        if not targets:
            return np.zeros((self.width, self.height))

        distances = np.full((self.width, self.height), np.inf)
        available_positions = []

        for x, y in targets:
            closed[x, y] = 1
            distances[x, y] = 0
            available_positions.extend(self.accessible_neighbors(closed, (x, y)))

        while register < ((self.width+self.height) * 10) and available_positions:
            next_open_positions = []
            for x, y in available_positions:
                distances[x, y] = self.minimum_distance(distances, (x, y))
                next_open_positions.extend(self.accessible_neighbors(closed, (x, y)))
            available_positions = next_open_positions
            register = register + 1

        for x, y in available_positions:
            distances[x, y] = register

        self.max_distance = register

        return distances

    """         FMM         """
    def fmm_update_target_grid(self):
        """
        Calculate the shortest distance from each grid point to the nearest target cell.
        This does not take barriers into account.
        :returns: The distance for every grid cell.
        """
        # print('FMM')
        mask = np.zeros((self.width, self.height))
        distances = np.ones((self.width, self.height))

        targets = [(x, y) for x in range(self.width) for y in range(self.height) if
                   self.grid[x, y] == Scenario.NAME2ID['TARGET']]
        for x, y in targets:
            distances[x, y] = 0

        for x in range(self.width):
            for y in range(self.height):
                if self.grid[x, y] == Scenario.NAME2ID['OBSTACLE']:
                    mask[x, y] = 1

        if not targets:
            return np.zeros((self.width, self.height))

        distances = skfmm.distance(np.ma.MaskedArray(distances, mask))
        self.max_distance = distances.max()

        return distances.data

    def update_step_task1(self):
        """
        Updates the position of all pedestrians.
        This does not take obstacles or other pedestrians into account.
        Pedestrians can occupy the same cell.
        """
        for pedestrian in self.pedestrians:
            pedestrian.update_step_task1(self)

    def update_step(self):
        """
        Updates the position of all pedestrians.
        This does not take obstacles or other pedestrians into account.
        Pedestrians can occupy the same cell.
        """
        scenario_update_position = False
        for pedestrian in self.pedestrians:
            scenario_update_move = pedestrian.update_step(self)
            if scenario_update_move is True:
                scenario_update_position = True

        return scenario_update_position


    @staticmethod
    def cell_to_color(_id):
        return Scenario.NAME2COLOR[Scenario.ID2NAME[_id]]

    def target_grid_to_image(self, canvas, old_image_id):
        """
        Creates a colored image based on the distance to the target stored in
        self.target_distance_gids.
        :param canvas: the canvas that holds the image.
        :param old_image_id: the id of the old grid image.
        """
        im = Image.new(mode="RGB", size=(self.width, self.height))
        pix = im.load()  # create the pixel map
        for x in range(self.width):
            for y in range(self.height):
                target_distance = self.target_distance_grids[x][y]
                pix[x, y] = (max(0, min(255, int(10 * target_distance) - 0 * 255)),
                             max(0, min(255, int(10 * target_distance) - 1 * 255)),
                             max(0, min(255, int(10 * target_distance) - 2 * 255)))
        im = im.resize(Scenario.GRID_SIZE, Image.NONE)
        self.grid_image = ImageTk.PhotoImage(im)
        canvas.itemconfigure(old_image_id, image=self.grid_image)

    def to_image(self, canvas, old_image_id):
        """
        Creates a colored image based on the ids stored in self.grid.
        Pedestrians are drawn afterwards, separately.
        :param canvas: the canvas that holds the image.
        :param old_image_id: the id of the old grid image.
        """

        im = Image.new(mode="RGB", size=(self.width, self.height)) #this is a Python Image object right now.

        # print(f"the width is {self.width} and the height is {self.height}")

        pix = im.load()  # create the pixel map
        for x in range(self.width):
            for y in range(self.height):
                pix[x, y] = Scenario.cell_to_color(self.grid[x, y])  #since it is a static method, using the class to access it.
        for pedestrian in self.pedestrians:
            x, y = pedestrian.position
            pix[x, y] = Scenario.NAME2COLOR['PEDESTRIAN']
        im = im.resize(Scenario.GRID_SIZE, Image.NONE)

        self.grid_image = ImageTk.PhotoImage(im) #this is a tkinter supported PhotoImage object which can be displayed on the canvas.

        canvas.itemconfigure(old_image_id, image=self.grid_image)


