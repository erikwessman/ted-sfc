import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_grid(frame, grid_size, skip_top_rows, skip_bottom_rows, skip_right_columns):
    h, w = frame.shape[:2]
    start_y = skip_top_rows * grid_size
    end_y = h - (h % grid_size) - skip_bottom_rows * grid_size
    end_x = w - (w % grid_size) - skip_right_columns * grid_size  # Adjusted for skipping columns

    # Draw vertical lines
    for x in range(0, end_x, grid_size):
        cv2.line(frame, (x, start_y), (x, end_y), (255, 0, 0), 1)

    # Draw horizontal lines
    for y in range(start_y, end_y, grid_size):
        cv2.line(frame, (0, y), (end_x, y), (255, 0, 0), 1)

    # Draw bottom horizontal line if within frame height
    if end_y + grid_size <= h:
        cv2.line(frame, (0, end_y), (end_x, end_y), (255, 0, 0), 1)

    # Draw rightmost vertical line if within frame width
    if end_x + grid_size <= w:
        cv2.line(frame, (end_x, start_y), (end_x, end_y), (255, 0, 0), 1)
# Video path and parameters
video_path = './data/c.webm'
output_path = './output/OF/c_grid.avi'
grid_size = 200
skip_top_rows = 3
skip_bottom_rows = 3
skip_right_columns = 5
scale = 10
thickness = 3
average_x_values = []
average_y_values = []
average_directions = []  
angle_diff_avg_vector_list = []
cell_angles = {cell_index: [] for cell_index in range(12)}
common_color = (255, 255, 255)  # White, for example
# Define a list of 10 different colors
colors = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
    (0, 128, 255),  # Orange
    (128, 0, 255),  # Pink
    (0, 255, 128),  # Lime
    (255, 128, 0),   # Sky Blue
    (128, 128, 128),  # Gray
    (128, 0, 0),      # Maroon
    (128, 128, 0),    # Olive
    (0, 128, 0),      # Dark Green
    (128, 0, 128)    # Purple
]

n = 4  # Number of frames to consider for moving average
angle_differences = {cell_index: [] for cell_index in range(12)}


second_cell_x = grid_size  # This is one cell width to the right of the first cell
second_cell_y = skip_top_rows * grid_size


# Open video
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    print("Unable to read video")
    cap.release()
    exit(0)


h, w = frame.shape[:2]
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))

first_cell_x = 0
first_cell_y = skip_top_rows * grid_size
frame_number = 0
avg_vector_angles = []
avg_angles = []


mean_flow_history = {cell_index: [] for cell_index in range(12)}  
angle_threshold = 114  #

def plot_angle_diff_peaks(cell_index, angle_diffs, threshold):
    # Find the indices (frames) where the clamped angle difference is non-zero
    peak_indices = [i for i, diff in enumerate(angle_diffs) if abs(diff) >= threshold]
    peak_values = [angle_diffs[i] for i in peak_indices]

    plt.figure()
    plt.plot(angle_diffs, label=f'Cell {cell_index + 1} Angle Differences')
    plt.xlabel('Frame')
    plt.ylabel('Angle Difference (degrees)')
    plt.title(f'Cell {cell_index + 1} Angle Differences Over Time')

    # Annotate the peaks
    for frame, angle_diff in zip(peak_indices, peak_values):
        plt.annotate(f'{angle_diff:.2f}\nFrame: {frame+4}',
                    xy=(frame - 1, angle_diff),  # Subtract 1 here if frame starts at 1
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=4,
                    arrowprops=dict(arrowstyle="->", color='red'))

    plt.legend()

def angle_to_vector(angle_degrees, scale=1):
                    """
                    Convert an angle in degrees to a vector.
                    """
                    angle_radians = np.radians(angle_degrees)
                    x = np.cos(angle_radians) * scale
                    y = np.sin(angle_radians) * scale
                    return (x, y)

angle_diff_over_time = {cell_index: [] for cell_index in range(12)}  
angle_differences_over_time = {cell_index: [] for cell_index in range(12)}  
angle_differences_per_frame = []

while True:
    sum_vectors = np.array([0.0, 0.0])
    count_vectors = 0
    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1  
    current_frame_differences = [frame_number]  

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prev_gray = gray

    
    vector_sum = np.zeros(2)
    cell_count = 0

    
    arrow_color = (0, 0, 255)  
    cell_index = 0  

    for y in range(skip_top_rows * grid_size, h - skip_bottom_rows * grid_size, grid_size):
        for x in range(0, w - skip_right_columns * grid_size, grid_size):
            # Define the cell boundaries
            cell_start_x = x
            cell_end_x = x + grid_size
            cell_start_y = y
            cell_end_y = y + grid_size

            # Ensure the cell is within the image boundaries
            cell_end_x = min(cell_end_x, w)
            cell_end_y = min(cell_end_y, h)
            # Calculate the text position (near the top center of each cell)
            text_position = (x + grid_size // 4, y + grid_size // 4)  # Adjust as needed for proper alignment

            # Compute mean flow vector for the cell
            cell_flow = flow[y:y+grid_size, x:x+grid_size]
            mean_flow = np.mean(cell_flow, axis=(0, 1))

            ## Compute the direction (angle) of the mean flow vector
            angle_radians = np.arctan2(mean_flow[1], mean_flow[0])
            angle_degrees = np.degrees(angle_radians)
            if angle_degrees < 0:
                angle_degrees += 360  # Normalize angle

             # Accumulate vectors
            vector_sum += mean_flow
            cell_count += 1

            if cell_index < 12:
                arrow_color = colors[cell_index]  # Use a color from the list
                cell_angles[cell_index].append(angle_degrees)

                if 6 <= cell_index <= 6:
                    arrow_color = common_color
                    sum_vectors += mean_flow
                    count_vectors += 1
                else:
                    arrow_color = colors[cell_index]  # Use the assigned color for other cells
                

                #moving_avg_vector = None
                moving_avg = None
                # Calculate moving average of the last n angles
                if len(cell_angles[cell_index]) >= n:
                    moving_avg = np.mean(cell_angles[cell_index][-n-1:-1])
                    if moving_avg < 0:
                        moving_avg += 360  # Normalize angle

                    ### Normalize the angle difference to be between 0 and -180 degrees
                if moving_avg is not None:
                    angle_diff = angle_degrees - moving_avg
                    #if angle_diff < 0:
                        #angle_diff += 360 
                    #if -360 <= angle_diff < -180:
                        #angle_diff += 180  # Adjusts angle_diff to be between 0 and 180
                    #elif angle_diff > 180:
                       # angle_diff -= 180  # Adjusts angle_diff to be between -180 and 0
                    #elif angle_diff < -180:
                       # angle_diff += 180  # Adjusts angle_diff to be between 0 and 180

                    #if angle_diff > 180:
                        #angle_diff -= 360  # Normalize angle difference to the range [-180, 180]
                    #elif angle_diff < -180:
                        #angle_diff += 360  # Normalize angle difference to the range [-180, 180]
                    # Adjusts angle_diff to be between 0 and 180
                    if -360 <= angle_diff < -180:
                        angle_diff += 180
                    else:
                        angle_diff = 0

                    # Clamp the angle difference if it's less than the threshold
                    clamped_angle_diff = -angle_diff if (-angle_diff) >= angle_threshold else 0
                    moving_avg_vector = angle_to_vector(moving_avg, scale)

                    # Store the clamped angle difference
                    angle_differences[cell_index].append(clamped_angle_diff)
                    if clamped_angle_diff != 0:
    # This means clamped_angle_diff is -angle_diff and meets the threshold condition
    # Draw a bounding box around the cell
                        cv2.rectangle(frame, (cell_start_x, cell_start_y), (cell_end_x, cell_end_y), (0, 255, 0), 2)  # Green bounding box


                    

                #mean_flow_history[cell_index].append(mean_flow)

                ## Calculate moving mean vector if we have n past vectors
                #if len(mean_flow_history[cell_index]) == n:
                    #moving_mean_flow = np.mean(mean_flow_history[cell_index], axis=0)
                #elif len(mean_flow_history[cell_index]) < n and len(mean_flow_history[cell_index]) > 0:
                    # If we have fewer than n vectors, use all available vectors to calculate the moving mean
                    #moving_mean_flow = np.mean(mean_flow_history[cell_index], axis=0)

                # Ensure the history doesn't exceed n entries
                    #if len(mean_flow_history[cell_index]) > n:
                        #mean_flow_history[cell_index].pop(0)

                ## Calculate the angle of the moving mean vector
                #moving_mean_angle = np.arctan2(moving_mean_flow[1], moving_mean_flow[0])
                #moving_mean_angle_degrees = np.degrees(moving_mean_angle)
                #if moving_mean_angle_degrees < 0:
                    #moving_mean_angle_degrees += 360  # Normalize angle
                    
                ## Compute the angle difference
                #if len(mean_flow_history[cell_index]) == n:
                    #angle_difff = angle_degrees- moving_mean_angle_degrees
                    # Clamp the angle difference if it's less than the threshold
                    #clamped_angle_difff = -angle_difff if (-angle_difff) >= angle_threshold else 0
                    #angle_diff_over_time[cell_index].append(clamped_angle_difff)
                    #print(f"Cell {cell_index}: Current Angle = {angle_degrees:.2f}, Moving Mean Angle = {moving_mean_angle_degrees:.2f}, Angle Difference = {angle_difff:.2f} degrees")
                
            
            
                    # Print the values
                    #print(f"Frame: {frame_number}, Cell: {cell_index + 1}, Angle Diff: {angle_diff:.2f}")
                    #print(f"Frame: {frame_number}, Cell: {cell_index + 1}, Angle: {angle_degrees:.2f}, Moving Avg: {moving_avg:.2f}, Angle Diff: {angle_diff:.2f}, Color: {colors[cell_index]}")
                    current_frame_differences.append(clamped_angle_diff)  # Append angle difference to list

                cell_index += 1
            else:
                arrow_color = (0, 0, 0)  # Default color for cells beyond the first 10

            # Draw the mean vector at the center of the cell
            center_x, center_y = x + grid_size // 2, y + grid_size // 2
            end_x = int(center_x + mean_flow[0] * scale)
            end_y = int(center_y + mean_flow[1] * scale)
            cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), arrow_color, thickness)

            # Draw the moving mean vector
            #moving_mean_end_x = int(center_x + moving_mean_flow[0] * scale)
            #moving_mean_end_y = int(center_y + moving_mean_flow[1] * scale)
            #cv2.arrowedLine(frame, (center_x, center_y), (moving_mean_end_x, moving_mean_end_y), (0, 255, 0), thickness)  # Green for moving mean vector

            
            #if moving_avg_vector is not None:
                    # Draw the moving average vector
                    #center_x, center_y = x + grid_size // 2, y + grid_size // 2
                    #end_x = int(center_x + moving_avg_vector[0])
                   # end_y = int(center_y + moving_avg_vector[1])
                   # cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), (0, 255, 255), thickness)  # Yellow color for avg vector

                

        if cell_index >= 12:
            break  # Exit the loop after processing the first 10 cells

    angle_differences_per_frame.append(current_frame_differences)

    # Compute and draw the average vector for cells 2, 3, 4, and 5 
    #if count_vectors > 0:
     #   avg_vector = sum_vectors / count_vectors
      #  avg_center_x = w // 2  # or any other logic for x position
       # avg_center_y = (skip_top_rows + 2) * grid_size  # or any other logic for y position
        #avg_end_x = int(avg_center_x + avg_vector[0] * scale)
        #avg_end_y = int(avg_center_y + avg_vector[1] * scale)
        #cv2.arrowedLine(frame, (avg_center_x, avg_center_y), (avg_end_x, avg_end_y), common_color, thickness)

        # Compute the direction (angle) of the average vector
        #avg_angle_radians = np.arctan2(avg_vector[1], avg_vector[0])
        #avg_angle_degrees = np.degrees(avg_angle_radians)
        #if avg_angle_degrees < 0:
         #   avg_angle_degrees += 360

        # Store the average angle
       # avg_vector_angles.append(avg_angle_degrees)

        # Check if there are enough angles to compute the moving average
        #if len(avg_vector_angles) >= n:
           # moving_avg_avg_vector = np.mean(avg_vector_angles[-n-1:-1]) 
            # Calculate the angle difference for the average vector
            #angle_diff_avg_vector = avg_angle_degrees - moving_avg_avg_vector
            # You can store this value or print it, as needed

            # Store the angle difference for the average vector
            #angle_diff_avg_vector_list.append(abs(angle_diff_avg_vector))


            # Print the values for the moving average of the average vector
            #print(f"Frame: {frame_number}, Average Vector Moving Avg: {moving_avg_avg_vector:.2f}, Angle Diff: {angle_diff_avg_vector:.2f}")  
             
             # Optionally, you can print out the average angle here
            #print(f"Frame: {frame_number}, Average Angle: {avg_angle_degrees:.2f}")


     # Display the frame number on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_position = (10, 30)  # You can adjust the position as needed
    font_scale = 1
    font_color = (255, 255, 255)  # White color
    line_type = 2

    cv2.putText(frame, f'Frame: {frame_number}', 
                text_position, 
                font, 
                font_scale, 
                font_color, 
                line_type)
        
# Draw the grid
    draw_grid(frame, grid_size, skip_top_rows=3, skip_bottom_rows=2, skip_right_columns=4)

    # Write frame to output video
    out.write(frame)

    # Show the frame
    cv2.imshow('Optical Flow with Grid', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

import csv

# Specify the CSV file name
csv_filename = './output/OF/angle_differences.csv'

# Write data to CSV
with open(csv_filename, 'w', newline='') as file:
    writer = csv.writer(file, delimiter=';')
    
    # Write the header
    header = ['frame_id'] + [f'cell{i+1}' for i in range(6)]  # Adjust the number of cells if necessary
    writer.writerow(header)
    
    # Write the data
    for frame_data in angle_differences_per_frame:
        writer.writerow(frame_data)


# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()


# Plotting the angle differences over time for each cell
#for cell_index in range(12):
    #plt.figure()
    #plt.plot(angle_differences[cell_index])
    #plt.title(f'Angle Difference Over Time for Cell {cell_index + 1}')
    #plt.xlabel('Frame')
    #plt.ylabel('Angle Difference (degrees)')

# After processing the video, plot the peaks for each cell
for cell_index in range(12):
    plot_angle_diff_peaks(cell_index, angle_differences[cell_index], angle_threshold)

# Add vertical lines at specific frames
    plt.axvline(x=177, color='red', linestyle='--', label='Frame 177')
    plt.axvline(x=223, color='red', linestyle='--', label='Frame 223')

    # Set custom tick labels for these specific frames on the x-axis
    plt.xticks([0, 177, 223, len(angle_differences[cell_index])], ['1', '177', '223', str(len(angle_differences[cell_index]))])

plt.show()



