import serial
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import sys

def get_next_index(folder_name):
    """Finds the next available index for new readings."""
    existing_files = [f for f in os.listdir(folder_name) if f.endswith('.png')]
    indices = [int(f.split('.')[0]) for f in existing_files if f.split('.')[0].isdigit()]
    return max(indices, default=-1) + 1

def save_matrix_as_image(matrix, folder_name, filename):
    """Saves an 8x8 NumPy matrix as a grayscale PNG image without axes and with tight layout."""
    plt.figure(figsize=(0.8, 0.8), dpi=100)
    plt.imshow(matrix, cmap='gray', vmin=0, vmax=500)
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(os.path.join(folder_name, filename), bbox_inches='tight', pad_inches=0)
    plt.close()

# Function to read 8 rows of data via serial communication after receiving "Print"
def read_matrix_from_serial(serial_port, folder_name, num_of_readings):
    folder_name = "trainingFiles/" + folder_name
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Load existing JSON data if available
    json_file = os.path.join(folder_name, 'matrix_data.json')
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data_dict = json.load(f)
    else:
        data_dict = {}
    
    ser = serial.Serial(serial_port, baudrate=115200, timeout=1)
    
    # # Set up the plot for real-time updates
    # plt.ion()  # Turn on interactive mode
    # fig, ax = plt.subplots()
    # img = ax.imshow(np.zeros((8, 8)), cmap='gray', vmin=0, vmax=500)  # Initialize with an empty matrix

    reading_count = 0  # Counter for the number of readings
    next_index = get_next_index(folder_name)    # Get the next available index
    
    while reading_count < num_of_readings:
        # Wait for the "Print" message to start reading a new matrix
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            print(f"{line}")
            
            if line[0] == "P":
                print("Starting to read new matrix...")
                rows = []  # Initialize an empty list to store rows
                
                # Read 8 rows for the matrix
                while len(rows) < 8:
                    if ser.in_waiting > 0:
                        row_line = ser.readline().decode('utf-8').strip()
                        row = list(map(int, row_line.split(',')))  # Convert row to a list of integers
                        print(row)
                        
                        if len(row) == 8:  # Ensure it's a valid row
                            rows.append(row)
                
                # Convert the list of rows into a NumPy array (our 8x8 matrix)
                matrix = np.array(rows)
                
                # Save the matrix data into the JSON file
                data_dict[reading_count + next_index] = matrix.tolist()

                # # Update the image data with the new matrix
                # img.set_data(matrix)
                
                # Save the plot as an image file
                img_file = os.path.join(folder_name, f'{reading_count + next_index}.png')
                save_matrix_as_image(matrix, folder_name, f'{reading_count + next_index}.png')
                # plt.savefig(img_file)
                print(f"Saved image: {img_file}")
                
                # # Redraw the canvas to update the plot in real-time
                # fig.canvas.draw()
                # fig.canvas.flush_events()
                
                reading_count += 1  # Increment the reading count

    # Save updated JSON
    with open(json_file, 'w') as json_f:
        json.dump(data_dict, json_f, indent=4)
    print(f"Saved JSON data: {json_file}")

# Main function to continuously read and plot matrices
def main():
    # Check for correct number of arguments
    if len(sys.argv) != 3:
        print("Usage: python readPlotAndSave.py <folder_name> <num_of_readings>")
        sys.exit(1)
    
    folder_name = sys.argv[1]  # First argument is the folder name
    num_of_readings = int(sys.argv[2])  # Second argument is the number of readings

    # Replace serial_port with your actual serial port (e.g., '/dev/ttyUSB0' on Linux or 'COM3' on Windows)
    serial_port = '/dev/tty.usbmodem101'
    
    # Continuously read and plot new matrices from serial communication
    read_matrix_from_serial(serial_port, folder_name, num_of_readings)

if __name__ == "__main__":
    main()
