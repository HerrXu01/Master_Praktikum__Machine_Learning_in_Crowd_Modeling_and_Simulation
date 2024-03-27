import sys
import tkinter
from tkinter import Button, Canvas, Menu, messagebox, ttk
from scenario_elements import Scenario, Pedestrian
import matplotlib.pyplot as plt
import json
import ast
import numpy as np
import time
import threading
import random
import csv

class MainGUI():
    """
    Defines a simple graphical user interface.
    To start, use the `start_gui` method.
    """
    def get_scenario(self):
        """
        Get the scenario parameters from the json file.

        Returns:
            data: A dictionary containing scenario parameters loaded from a JSON file.
        """
        with open('scenario.json', "r") as file:
            data =  json.load(file)
        return data

    def save_input_parameters(self, input_window, scenario, data, canvas, canvas_image):
        """
        Save the parameters that the user input and store them in a json file.
        Then visualize the new scenario with the new parameters.

        Args:
            input_window: The input window where the user entered parameters.
            scenario (scenario_elements.Scenario): The scenario instance to be updated.
            data (dict): A dictionary containing user-entered parameters.
            canvas (tkinter.Canvas): The canvas used for visualization.
            canvas_image: The image on the canvas for displaying the scenario.
        """
        for key, value in data["automata_size"].items():
            data["automata_size"][key] = int(value.get())


        data["targets"] = ast.literal_eval(data["targets"].get())
        data["obstacles"] = ast.literal_eval(data["obstacles"].get())

        pds = []
        pds_position = ast.literal_eval(data["pedestrians"][0].get())
        pds_speeds = ast.literal_eval(data["pedestrians"][1].get())
        for i in range(len(pds_speeds)):
            pds.append({"pos": pds_position[i], "speed": pds_speeds[i]})
        data["pedestrians"] = pds

        with open("scenario.json", "w") as json_file:
            json.dump(data, json_file)

        input_window.destroy()
        
        scenario.width = data["automata_size"]["width"]
        scenario.height = data["automata_size"]["height"]
        scenario.grid = np.zeros((scenario.width, scenario.height))
        scenario.pedestrians = [Pedestrian((pedestrian["pos"][0], pedestrian["pos"][1]), pedestrian["speed"]) for pedestrian in data["pedestrians"]]
        
        for target in data["targets"]:
            scenario.grid[target[0], target[1]] = Scenario.NAME2ID['TARGET']
        if len(data["obstacles"][0]) > 0:
            for obstacle in data["obstacles"]:
                scenario.grid[obstacle[0], obstacle[1]] = Scenario.NAME2ID['OBSTACLE']
        scenario.recompute_target_distances()
        scenario.to_image(canvas, canvas_image)
        

    def create_scenario(self, win, scenario, canvas, canvas_image):
        """
        Open a new window, and enable the users to create a new scenario.

        Args:
            win: The main window of the GUI.
            scenario (scenario_elements.Scenario): The scenario instance to be updated.
            canvas (tkinter.Canvas): The canvas used for visualization.
            canvas_image: The image on the canvas for displaying the scenario.
        """
        input_window = tkinter.Toplevel(win)
        input_window.title("Input Parameters")
        input_window.geometry('650x650')

        data = self.get_scenario()
        # Get automata size
        for item in data["automata_size"]:
            label = tkinter.Label(input_window, text=f"Automata {item} (positive integer): ")
            label.pack()
            automata_size_entry = tkinter.Entry(input_window, width=5)
            automata_size_entry.pack()
            data["automata_size"][item] = automata_size_entry

        # Get targets' positions
        label_target = tkinter.Label(input_window, text="Positions of Targets (in the form of list of lists):")
        label_target.pack()
        reminder = tkinter.Label(input_window, text="e.g. [[10, 20], [30, 40]]")
        reminder.pack()
        target_entry = tkinter.Entry(input_window, width=30)
        target_entry.pack()
        data["targets"] = target_entry

        # Get obstacles' positions
        label_obs = tkinter.Label(input_window, text="Positions of Obstacles (in the form of list of lists):")
        label_obs.pack()
        reminder = tkinter.Label(input_window, text="e.g. [[10, 20], [30, 40]]")
        reminder.pack()
        obs_entry = tkinter.Entry(input_window, width=30)
        obs_entry.pack()
        data["obstacles"] = obs_entry

        # Get the positions and the speeds of pedestrians
        label_pds = tkinter.Label(input_window, text="Positions of Pedestrians (in the form of list of lists):")
        label_pds.pack()
        reminder = tkinter.Label(input_window, text="e.g. [[10, 20], [30, 40]]")
        reminder.pack()
        pds_entry = tkinter.Entry(input_window, width=30)
        pds_entry.pack()
        label_speeds = tkinter.Label(input_window, text="Speeds of Pedestrians (in the form of list):")
        label_speeds.pack()
        reminder = tkinter.Label(input_window, text="e.g. [1, 2]. Note: the length must be equal to the number of pedestrians!")
        reminder.pack()
        speeds_entry = tkinter.Entry(input_window, width=30)
        speeds_entry.pack()
        data["pedestrians"] = (pds_entry, speeds_entry)


        apply_button = tkinter.Button(input_window, text="Apply", command=lambda: self.save_input_parameters(input_window, scenario, data, canvas, canvas_image))
        apply_button.pack()


    def restart_scenario(self, scenario, canvas, canvas_image):

        """
        Restart the scenario using the initial parameters and visualize it.

        Args:
            scenario (scenario_elements.Scenario): The scenario instance to be restarted.
            canvas (tkinter.Canvas): The canvas used for visualization.
            canvas_image: The image on the canvas for displaying the scenario.
        """
        data = self.get_scenario()
        scenario.pedestrians = [Pedestrian((pedestrian["pos"][0], pedestrian["pos"][1]), pedestrian["speed"]) for pedestrian in data["pedestrians"]]
        scenario.to_image(canvas, canvas_image)

    def step_scenario_task1(self, scenario, canvas, canvas_image):
        """
        Moves the simulation forward by one step, and visualizes the result.

        Args:
            scenario (scenario_elements.Scenario): The scenario instance to be updated.
            canvas (tkinter.Canvas): The canvas used for visualization.
            canvas_image: The image on the canvas for displaying the scenario.
        """
        scenario.update_step_task1()
        scenario.to_image(canvas, canvas_image)

    def step_scenario(self, scenario, canvas, canvas_image):
        """
        Moves the simulation forward by one step, and visualizes the result.

        Args:
            scenario (scenario_elements.Scenario): The scenario instance to be updated.
            canvas (tkinter.Canvas): The canvas used for visualization.
            canvas_image: The image on the canvas for displaying the scenario.
        """
        scenario.update_step()
        scenario.to_image(canvas, canvas_image)

    
    def automated_step(self, scenario, canvas, canvas_image, data):
        """
        Automatically moves the pedestrians forward to the targets, and visualizes the result.

        Args:
            scenario (scenario_elements.Scenario): The scenario instance to be updated.
            canvas (tkinter.Canvas): The canvas used for visualization.
            canvas_image: The image on the canvas for displaying the scenario.
            data (dict): A dictionary containing user-entered parameters.
        """
        
        # print("me")
        def run_simulation():
            while True:
                self.step_scenario_task1(scenario, canvas, canvas_image)
                time.sleep(0.3)
                check = []
                for pedestrian in scenario.pedestrians:
                    check.append([pedestrian.position[0], pedestrian.position[1]] in data["targets"])

                if all(check):
                    break
        
        simulation_thread = threading.Thread(target=run_simulation)
        simulation_thread.start()

    def automated_step_task51(self, scenario, canvas, canvas_image, targets):
        """
        Task 1, Scenario 1 RiMea: Automatically moves the pedestrians forward to the targets, and visualizes the result.

        Args:
            scenario (scenario_elements.Scenario): The scenario instance to be updated.
            canvas (tkinter.Canvas): The canvas used for visualization.
            canvas_image: The image on the canvas for displaying the scenario.
            targets: List of target coordinates.
        """
        
        def run_simulation():
            travel_time = 0
            while True:
                self.step_scenario(scenario, canvas, canvas_image)
                travel_time = travel_time + 1
                time.sleep(0.3)
                check = []
                for pedestrian in scenario.pedestrians:
                    check.append([pedestrian.position[0], pedestrian.position[1]] in targets)
                if all(check):
                    break
            
            print("total travel time for the pedestrian is ",travel_time)
        
        simulation_thread = threading.Thread(target=run_simulation)
        simulation_thread.start()

    def automated_step_task54(self, scenario, canvas, canvas_image, targets):
        """
        Automatically moves the pedestrians forward to the targets, and visualizes the result.

        Args:
            scenario (scenario_elements.Scenario): The scenario instance to be updated.
            canvas (tkinter.Canvas): The canvas used for visualization.
            canvas_image: The image on the canvas for displaying the scenario.
            targets: List of target coordinates.
        """
        #scenario.pdestrians.clear()
        scenario.pedestrian = []
        def run_simulation():
            tim = np.zeros(50)
            travel_tim = np.zeros(50)
            ped_age = np.zeros(50)
            ped_speed = np.zeros(50)
            ped_dist = np.zeros(50)
            ped_cal_speed = np.zeros(50)
            
            #Loading Weidmann Input Data file
            with open('speed_age_dist.json', "r") as file:
                weidmann_data =  json.load(file)

            data = [(pedestrian["Age"], pedestrian["Mean_speed"], pedestrian["Speed_deviation"])for pedestrian in weidmann_data["pedestrians"]]  

            #Considering 50 pedestrians - 10 each of 20, 30, 40, 50, 60 years of age and their corresponding speed in accordance with Weidmann graph
            #Speed value for every pedestrian random sampled using its mean and standard deviation
            for i in range(50):
                ped_age[i] = data[i//10][0]
                ped_speed[i] = data[i//10][1] + data[i//10][2]*np.random.uniform(-1,1)

            #Random y coordinate start from Starting plane X=14
            random_ystart =  random.choices(range(39,60), k=50)
            
            simulation_time = 0
            ped_count = 0
            prev_check = []
            prev_ped_pos = []

            #Generating total 50 pedestrians one after every three simulation steps to mitigate congesation
            while True:  
                if (simulation_time%3 == 0) and ped_count < 50:
                    ped = Pedestrian((14, random_ystart[ped_count]), ped_speed[ped_count])
                    scenario.pedestrians.append(ped)
                    ped_count+=1
                    prev_check.append(False)

                self.step_scenario(scenario, canvas, canvas_image)
                tim[:ped_count]+= 1
                simulation_time = simulation_time + 1

                time.sleep(0.3)
                check = []
                curr_ped_pos = []

                #Updating current position and checking target reaching condition for each pedestrian
                for pedestrian in scenario.pedestrians:
                    check.append([pedestrian.position[0], pedestrian.position[1]] in targets)
                    curr_ped_pos.append(pedestrian.position)

                #Updating distance travelled for each pedestrian
                for i in range(len(prev_ped_pos)):
                    ped_dist[i] += np.sqrt((prev_ped_pos[i][0]-curr_ped_pos[i][0])**2+(prev_ped_pos[i][1]-curr_ped_pos[i][1])**2)

                prev_ped_pos = curr_ped_pos

                #Calculating total travel time and speed for target reaching pedestrian
                ii = (i for i, v in enumerate(prev_check) if v != check[i])
                for i, v in enumerate(prev_check):
                    if v != check[i]:
                        travel_tim[i] = tim[i]
                        ped_cal_speed[i] = ped_dist[i]/travel_tim[i]

                prev_check = check

                #Writing parameters[Age, Distance, Travel_Time, Observed_Speed] to CSV file and generating plot to demonstrate validation of Weidmann Age-Speed data
                if all(check):
                    with open("age_speed_dist.csv","w") as f:
                        writer = csv.writer(f, delimiter='\t')
                        writer.writerows(zip(ped_age.tolist(),ped_dist.tolist(),travel_tim.tolist(),ped_cal_speed.tolist()))
                        plt.scatter(ped_age,ped_cal_speed)
                        plt.title('Age vs Observed Horizontal Walking Speed')
                        plt.xlabel("Age(in years)")
                        plt.ylabel("Walking_Speed(m/s)")
                        plt.show()
                    break

        simulation_thread = threading.Thread(target=run_simulation)
        simulation_thread.start()

    def new_automated_step(self, scenario, canvas, canvas_image, data):
        # print("ME")
        """
        An extended version of the automated_step function which takes care of speeds.\n
        This function treats (horizontal, vertical) neighbours and diagonal neighbours differently.\n
        Pedestrian steps to these different neighbour cells in different time intervals, thus maintaining a uniform speed across them.

        Args:
            scenario (scenario_elements.Scenario): The scenario instance to be updated.
            canvas (tkinter.Canvas): The canvas used for visualization.
            canvas_image: The image on the canvas for displaying the scenario.
            data (dict): A dictionary containing user-entered parameters.
        """
        def run_speed_simulation(pedestrian):

            while True:
                neighbors = pedestrian.get_neighbors(scenario)
                next_cell_distance = scenario.target_distance_grids[pedestrian._position[0]][pedestrian._position[1]]
                next_pos = pedestrian._position
                for (n_x, n_y) in neighbors:
                    if scenario.grid[n_x, n_y] == scenario.NAME2ID["OBSTACLE"] or scenario.grid[n_x, n_y] == scenario.NAME2ID["PEDESTRIAN"]:
                        continue
                    if next_cell_distance > scenario.target_distance_grids[n_x, n_y]:
                        next_pos = (n_x, n_y)
                        next_cell_distance = scenario.target_distance_grids[n_x, n_y]
                
                time_for_horizontal_vertical_neighbour = 1/pedestrian._desired_speed
                time_for_diagonal_neighbour = time_for_horizontal_vertical_neighbour * np.sqrt(2)
                
                if (next_pos[0] == pedestrian._position[0] or next_pos[1] == pedestrian._position[1]):
                    time.sleep(time_for_horizontal_vertical_neighbour)
                else:
                    time.sleep(time_for_diagonal_neighbour)
                
                pedestrian._position = next_pos
                scenario.to_image(canvas, canvas_image)

                if scenario.grid[pedestrian._position[0], pedestrian._position[1]] == scenario.NAME2ID["TARGET"]:

                    scenario.pedestrians.remove(pedestrian) #makes the targets absorbing
                    scenario.to_image(canvas, canvas_image)

                    messagebox.showinfo("Info", "Pedestrian succesfully reached nearest target.")

                    return

        for pedestrian in scenario.pedestrians:
            simulation_thread = threading.Thread(target=run_speed_simulation, args=(pedestrian,))
            simulation_thread.start()

    def automated_step_task52(self, scenario, canvas, canvas_image, measure_point, density, num_ped):
        """
        Automatically moves the pedestrians forward to the targets, and visualizes the result.

        Args:
            scenario (scenario_elements.Scenario): The scenario instance to be updated.
            canvas (tkinter.Canvas): The canvas used for visualization.
            canvas_image: The image on the canvas for displaying the scenario.
            measure_point: List of measurement points' coordinates.
            density: Density value of pedestrians.
            num_ped: Number of pedestrians.
        """
        
        def run_simulation():
            #initilizations
            ped_dist = np.zeros(num_ped)
            simulation_time = 0
            prev_ped_pos = []

            #It is simulating by step scenario in 70 simulation seconds
            while (simulation_time<=70):  
                self.step_scenario(scenario, canvas, canvas_image)
                simulation_time = simulation_time + 1

                time.sleep(0.3)
                check = []
                curr_ped_pos = []
                
                #Updating current position and checking target reaching condition for each pedestrian
                for pedestrian in scenario.pedestrians:
                    curr_ped_pos.append(pedestrian.position)

                #Updating distance travelled for each pedestrian
                for i in range(len(prev_ped_pos)):
                    ped_dist[i] += np.sqrt((prev_ped_pos[i][0]-curr_ped_pos[i][0])**2+(prev_ped_pos[i][1]-curr_ped_pos[i][1])**2)

                #current pedestrian position becomes previous pedestrian position
                prev_ped_pos = curr_ped_pos
            
            #Checking and getting the pedestrians in the measure points
            for pedestrian in scenario.pedestrians:
                check.append([pedestrian.position[0], pedestrian.position[1]] in measure_point)
            
            #getting the pedestrians in the measure point
            A = np.where(check)
            ped_avg_speed = np.zeros(len(A[0]))
            
            #calculating the average speed of pedestrian
            for i in range(len(A[0])):
                ped_avg_speed[i] = (ped_dist[A[0][i]])/70
            
            #calculating flux
            flux = np.mean(ped_avg_speed) * density
            print("Average speed is: ", np.mean(ped_avg_speed))
            print("Flux is:", flux)
        
        #running simulation            
        simulation_thread = threading.Thread(target=run_simulation)
        simulation_thread.start()

    def draw_obstacles(self, scenario, canvas, canvas_image):
        scenario.grid = np.zeros((scenario.width, scenario.height))
        for i in range(30):
            scenario.grid[10 + i, 60] = Scenario.NAME2ID['OBSTACLE']
            scenario.grid[60 + i, 60] = Scenario.NAME2ID['OBSTACLE']
            scenario.grid[60 + i, 59] = Scenario.NAME2ID['OBSTACLE']
            scenario.grid[10 + i, 90] = Scenario.NAME2ID['OBSTACLE']
            scenario.grid[60 + i, 90] = Scenario.NAME2ID['OBSTACLE']
            scenario.grid[60 + i, 91] = Scenario.NAME2ID['OBSTACLE']
            scenario.grid[10, 60 + i] = Scenario.NAME2ID['OBSTACLE']
            scenario.grid[89, 60 + i] = Scenario.NAME2ID['OBSTACLE']
        for i in range(31):
            scenario.grid[10 + i, 59] = Scenario.NAME2ID['OBSTACLE']
            scenario.grid[10 + i, 91] = Scenario.NAME2ID['OBSTACLE']
        for i in range(33):
            scenario.grid[9, 59 + i] = Scenario.NAME2ID['OBSTACLE']
            scenario.grid[90, 59 + i] = Scenario.NAME2ID['OBSTACLE']
        for i in range(13):
            scenario.grid[40, 60 + i] = Scenario.NAME2ID['OBSTACLE']
            scenario.grid[41, 59 + i] = Scenario.NAME2ID['OBSTACLE']

            scenario.grid[40, 78 + i] = Scenario.NAME2ID['OBSTACLE']
            scenario.grid[41, 79 + i] = Scenario.NAME2ID['OBSTACLE']

            scenario.grid[60, 60 + i] = Scenario.NAME2ID['OBSTACLE']
            scenario.grid[59, 59 + i] = Scenario.NAME2ID['OBSTACLE']

            scenario.grid[60, 78 + i] = Scenario.NAME2ID['OBSTACLE']
            scenario.grid[59, 79 + i] = Scenario.NAME2ID['OBSTACLE']

        for i in range(20):
            scenario.grid[40 + i, 72] = Scenario.NAME2ID['OBSTACLE']
            scenario.grid[41 + i, 71] = Scenario.NAME2ID['OBSTACLE']
            scenario.grid[40 + i, 78] = Scenario.NAME2ID['OBSTACLE']
            scenario.grid[41 + i, 79] = Scenario.NAME2ID['OBSTACLE']

        for i in range(30):
            scenario.grid[15 + i, 5] = Scenario.NAME2ID['OBSTACLE']
            scenario.grid[15 + i, 6] = Scenario.NAME2ID['OBSTACLE']
            scenario.grid[15 + i, 30] = Scenario.NAME2ID['OBSTACLE']
            scenario.grid[15 + i, 31] = Scenario.NAME2ID['OBSTACLE']
        for i in range(25):
            scenario.grid[44, 5 + i] = Scenario.NAME2ID['OBSTACLE']
        for i in range(27):
            scenario.grid[45, 5 + i] = Scenario.NAME2ID['OBSTACLE']
        for i in range(8):
            scenario.grid[89, 72 + i] = Scenario.NAME2ID['TARGET']
            scenario.grid[90, 72 + i] = Scenario.NAME2ID['TARGET']
            scenario.grid[76, 15 + i] = Scenario.NAME2ID['TARGET']

        scenario.pedestrians = self.filling_pedestrians((11, 61), (20, 29), 150, 1) + self.filling_pedestrians((1, 10),
                                                                                                               (30, 15),
                                                                                                               150, 1)
        # can be used to show pedestrians and targets
        scenario.to_image(canvas, canvas_image)
    
    def straight_path_task51(self, scenario, canvas, canvas_image):
        """
        Draws the straight path 40mx2m (modeled as obstacles) with end plane as target plane with 1 pedestrian 
        being generated for simulation randomly starting at Starting Plane [X=29]
        Speed of the pedestrians is uniformly randomly selected from distribution with given mean and variance
        Calls the automated function for Task 5.1 for running automated simulation
        """
        scenario.grid = np.zeros((scenario.width, scenario.height))
        for i in range(40):
            scenario.grid[30 + i, 48] = Scenario.NAME2ID['OBSTACLE']
            scenario.grid[30 + i, 51] = Scenario.NAME2ID['OBSTACLE']
    
        targets = []
        for i in range(4):
            scenario.grid[70, 48 + i] = Scenario.NAME2ID['TARGET']
            targets.append([70, 49 + i])

        speed = 1.33 + 0.05*np.random.uniform(-1,1)
        scenario.pedestrians = []
        scenario.pedestrians = self.filling_pedestrians((29, 50), (1, 2), 1, speed) 

        self.automated_step_task51(scenario, canvas, canvas_image, targets)

    def straight_path_task54(self, scenario, canvas, canvas_image):
        """
        Draws the straight path 70mx20m (modeled as obstacles) with end plane as target plane 
        Calls the automated function for Task 5.4 for running automated simulation
        """
        #scenario.pdestrian.clear()
        scenario.grid = np.zeros((scenario.width, scenario.height))

        for i in range(70):
            scenario.grid[15 + i, 39] = Scenario.NAME2ID['OBSTACLE']
            scenario.grid[15 + i, 60] = Scenario.NAME2ID['OBSTACLE']
            #obstacle.append(15 + i, 39)
            #obstacle.append(15 + i, 60)

        targets = []
        for i in range(22):
            scenario.grid[85, 39 + i] = Scenario.NAME2ID['TARGET']
            targets.append([85, 39 + i])
            #target.append([84, 40 + i])

        self.automated_step_task54(scenario, canvas, canvas_image, targets)

    def filling_pedestrians(self, start_position, size, pedestrians_num, pedestrians_speed):
        """
        Fill a given area in the scene with a given number of pedestrians distributed uniformly and randomly.
        :param start_position:tuple of the starting point of the area to be filled
        :param size:tuple of size of area to fill
        :param pedestrians_num:Number of pedestrians filled
        :param pedestrians_speed: the speed of filling pedestrians
        :return:
        """
        pedestrians = []
        for pos in random.sample(range(size[0] * size[1]), pedestrians_num):
            pedestrians.append(
                Pedestrian((start_position[0] + pos % size[0], start_position[1] + pos // size[0]), pedestrians_speed))
        return pedestrians

    def task_41(self, scenario, canvas, canvas_image):
        """
        Task4 uses dijkstra algorithm
        """
        self.draw_obstacles(scenario, canvas, canvas_image)
        scenario.compute_method = 0
        scenario.recompute_target_distances()

    def task_42(self, scenario, canvas, canvas_image):
        """
        Task4 uses FMM algorithm
        """
        self.draw_obstacles(scenario, canvas, canvas_image)
        scenario.compute_method = 1
        scenario.recompute_target_distances()

    def task_43(self, scenario, canvas, canvas_image):
        """
        Task4 uses basic algorithm
        """
        self.draw_obstacles(scenario, canvas, canvas_image)
        scenario.compute_method = 2
        scenario.recompute_target_distances()
    
    """TEST 3"""
    def draw_obstacles_for_test3(self, scenario, canvas, canvas_image):
        """
        This function drawing obstacles, targets and pedestrians for test 3

        Args:
            scenario (scenario_elements.Scenario): The scenario instance to be updated.
            canvas (tkinter.Canvas): The canvas used for visualization.
            canvas_image: The image on the canvas for displaying the scenario.
        """
        #creating grid
        scenario.grid = np.zeros((scenario.width, scenario.height))
        
        #creating obstacles
        for i in range(30):
            scenario.grid[59, 50 + i] = Scenario.NAME2ID['OBSTACLE']
            scenario.grid[30 + i, 80] = Scenario.NAME2ID['OBSTACLE']
        
        for i in range(25):
            scenario.grid[54, 50 + i] = Scenario.NAME2ID['OBSTACLE']
            scenario.grid[30 + i, 75] = Scenario.NAME2ID['OBSTACLE']
            
        for i in range(6):
            scenario.grid[30, 75 + i] = Scenario.NAME2ID['OBSTACLE']
        
        #creating targets
        for i in range(4):
            scenario.grid[55 + i, 50] = Scenario.NAME2ID['TARGET']
            
        #creating pedestrians
        #this line fill up the corridor with uniformly and randomly
        #between (31,76) - (46,80) , 20 pedestrians, pedestrian speed is 1    
        scenario.pedestrians = self.filling_pedestrians((31, 76), (15, 4), 20, 1)
        
        # can be used to show pedestrians and targets
        scenario.to_image(canvas, canvas_image)
        
    def test_3(self, scenario, canvas, canvas_image):
        """
        Test 3 
        Drawing obstacles, choosing compute method which 0=dijkstra,1=FMM,2=basic
        """
        self.draw_obstacles_for_test3(scenario, canvas, canvas_image)
        scenario.compute_method = 1
        scenario.recompute_target_distances()
        
    """TEST 2"""
    def draw_obstacles_for_test2(self, scenario, canvas, canvas_image):
        """
        This function drawing obstacles, targets and pedestrians for test 2

        Args:
            scenario (scenario_elements.Scenario): The scenario instance to be updated.
            canvas (tkinter.Canvas): The canvas used for visualization.
            canvas_image: The image on the canvas for displaying the scenario.
        """
        #These lines creates grid, obstacles, and targets
        
        scenario.grid = np.zeros((scenario.width, scenario.height))
        
        for i in range(998):
            scenario.grid[1 + i, 30] = Scenario.NAME2ID['OBSTACLE']
            scenario.grid[1 + i, 50] = Scenario.NAME2ID['OBSTACLE']
        
        for i in range(21):
            scenario.grid[999, 30 + i] = Scenario.NAME2ID['TARGET']
        
        
        """    
        Measurement areas:
        (490, 40) - (510, 60)
        (490, 60) - (510, 80)
        (440, 40) - (460, 60)
        """
        measure_point = []
        for i in range(20):
            for j in range(20):
                measure_point.append([490 + i, 40 + j])
                measure_point.append([490 + i, 60 + j])
                measure_point.append([440 + i, 40 + j])
        
        """
        for %density, % # of pedestrian
        for 0.5, 800
        for 1, 1600
        for 2, 3200
        for 3, 4800
        for 4, 6400
        for 5, 8000
        for 6, 9600
        """
        #user needs to change this density to make different measurements
        density = 2
        num_ped = 0
        if(density==0.5):
            num_ped = 800
        elif(density==1):
            num_ped = 1600
        elif(density==2):
            num_ped = 3200 
        elif(density==3):
            num_ped = 4800 
        elif(density==4):
            num_ped = 6400 
        elif(density==5):
            num_ped = 8000 
        elif(density==6):
            num_ped = 9600     
        
        #this line fill up the corridor with uniformly and randomly
        #between (1,31) - (999,50) , pedestrian speed is 1.3                       
        scenario.pedestrians = self.filling_pedestrians((1, 31), (998, 19), num_ped, 1.3)
        
        #this line making the automated step for test 2
        self.automated_step_task52(scenario, canvas, canvas_image, measure_point,density,num_ped)
    
    def test_2(self, scenario, canvas, canvas_image):
        """
        Test 2 
        Drawing obstacles, choosing compute method which 0=dijkstra,1=FMM,2=basic
        """
        self.draw_obstacles_for_test2(scenario, canvas, canvas_image)
        scenario.compute_method = 1
        scenario.recompute_target_distances()


    def task_51(self, scenario, canvas, canvas_image):
        """
        Task 5.1 to demonstrate Scenario 1: Straight line movement
        """
        self.straight_path_task51(scenario, canvas, canvas_image)
        scenario.compute_method = 2
        scenario.recompute_target_distances()
    
    def task_54(self, scenario, canvas, canvas_image):
        """
        Task 5.1 to demonstrate Scenario 7 of RiMea guidelines: Walking speed with age distribution
        """
        self.straight_path_task54(scenario, canvas, canvas_image)
        scenario.compute_method = 2
        scenario.recompute_target_distances()
        

    def exit_gui(self, ):
        """
        Close the GUI.
        """
        sys.exit()

    def start_gui(self, ):
        """
        Creates and shows a simple user interface with a menu and multiple buttons.
        Only one button works at the moment: "step simulation".
        Also creates a rudimentary, fixed Scenario instance with three Pedestrian instances and multiple targets.
        """
        win = tkinter.Tk()
        win.geometry('650x650')  # setting the size of the window
        win.title('Cellular Automata GUI')

        menu = Menu(win)
        win.config(menu=menu)
        file_menu = Menu(menu)
        menu.add_cascade(label='Simulation', menu=file_menu)
        file_menu.add_command(label='New', command=lambda: self.create_scenario(win, sc, canvas, canvas_image))
        file_menu.add_command(label='Restart', command=lambda: self.restart_scenario(sc, canvas, canvas_image))
        file_menu.add_command(label='Close', command=self.exit_gui)

        canvas = Canvas(win, width=Scenario.GRID_SIZE[0] * 2, height=Scenario.GRID_SIZE[1] * 3)  # creating the canvas

        canvas_image = canvas.create_image(75, 100, image=None, anchor=tkinter.NW)
        
        canvas.pack(side="right")


        data = self.get_scenario()
        
        sc = Scenario(data["automata_size"]["width"], data["automata_size"]["height"])

        for target in data["targets"]:
            sc.grid[target[0], target[1]] = Scenario.NAME2ID['TARGET']

        if len(data["obstacles"][0]) > 0:
            for obstacle in data["obstacles"]:
                sc.grid[obstacle[0], obstacle[1]] = Scenario.NAME2ID['OBSTACLE']
        
        sc.recompute_target_distances()

        sc.pedestrians = [Pedestrian((pedestrian["pos"][0], pedestrian["pos"][1]), pedestrian["speed"]) for pedestrian in data["pedestrians"]]

        # can be used to show pedestrians and targets
        sc.to_image(canvas, canvas_image)

        # can be used to show the target grid instead
        # sc.target_grid_to_image(canvas, canvas_image)

        btn_step = Button(win, text='Step simulation', command=lambda: self.step_scenario_task1(sc, canvas, canvas_image))
        btn_step.place(x=20, y=10)
        btn_automate = Button(win, text='Automated step', command=lambda: self.automated_step(sc, canvas, canvas_image, data))
        btn_automate.place(x=140, y=10)
        btn = Button(win, text='Restart simulation', command=lambda: self.restart_scenario(sc, canvas, canvas_image))
        btn.place(x=260, y=10)
        # btn["state"] = DISABLED
        btn = Button(win, text='Create simulation', command=lambda: self.create_scenario(win, sc, canvas, canvas_image))
        btn.place(x=380, y=10)
        btn = Button(win, text='Task 4(DIJKSTRA)', command=lambda: self.task_41(sc, canvas, canvas_image))
        btn.place(x=20, y=40)
        btn = Button(win, text='Task 4(FMM)', command=lambda: self.task_42(sc, canvas, canvas_image))
        btn.place(x=180, y=40)
        btn = Button(win, text='Task 4(BASIC)', command=lambda: self.task_43(sc, canvas, canvas_image))
        btn.place(x=310, y=40)
        btn = Button(win, text='TEST 1', command=lambda: self.task_51(sc, canvas, canvas_image))
        btn.place(x=20, y=70)
        btn = Button(win, text='TEST 2', command=lambda: self.test_2(sc, canvas, canvas_image))
        btn.place(x=140, y=70)
        btn = Button(win, text='TEST 3', command=lambda: self.test_3(sc, canvas, canvas_image))
        btn.place(x=260, y=70)
        btn = Button(win, text='TEST 4', command=lambda: self.task_54(sc, canvas, canvas_image))
        btn.place(x=380, y=70)

        speed_check = tkinter.IntVar()

        # A check button for simulations involving speeds.
        # To show the real effect of speeds in simulation, the step functionality is disabled as soon as speed_check = True
        # Also the functionality of btn_automate is replaced with the speed adjusted version of the automated_step function.
        def toggle_button(button):
            button.config(state=tkinter.DISABLED if speed_check.get() else tkinter.NORMAL)
        
        def toggle_function(button):
            button.config(command=lambda: self.new_automated_step(sc, canvas, canvas_image, data) if speed_check.get() else self.automated_step(sc, canvas, canvas_image, data))

        checkbutton = ttk.Checkbutton(win, variable=speed_check, text='Simulation \ninvolves speeds', command=lambda: [toggle_button(btn_step), toggle_function(btn_automate)])
        
        checkbutton.place(x=500, y=5)

        win.mainloop()
