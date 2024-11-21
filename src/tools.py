import os
import subprocess
import shutil

def generate_input_file(pipe_thickness):
    command = f"abaqus cae noGUI=./src/utils/create_inp_file.py -- {pipe_thickness}"
    try:
        subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("Input file generated successfully.")

        # Define the source and destination paths
        source_path = 'cantilever_beam.inp'
        destination_path = 'src/abaqus_files/cantilever_beam.inp'

        # Move the file
        shutil.move(source_path, destination_path)
        # print("File moved successfully.")

    except subprocess.CalledProcessError as e:
        print("Error during Abaqus job execution.")
        print("Error output:", e.stderr)

def run_abaqus():
    # Run the Abaqus job
    command = f"abaqus job=cantilever_beam input=src/abaqus_files/cantilever_beam.inp"
    try:
        subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("Abaqus job completed successfully.")
        
        # Define the source directory
        source_directory = os.getcwd()  # Assumes Abaqus outputs files in the current working directory

        # Loop through files in the source directory
        for file_name in os.listdir(source_directory):
            source_path = os.path.join(source_directory, file_name)
            if os.path.isfile(source_path) and file_name.startswith("cantilever_beam"):
                target_path = os.path.join("src/abaqus_files", file_name)
                shutil.move(source_path, target_path)
                # print(f"Moved {file_name} to 'src/abaqus_files'")
    except subprocess.CalledProcessError as e:
        print("Error during Abaqus job execution.")
        print("Error output:", e.stderr)

def extract_von_mises_stress_from_ODB():
    command = f"abaqus python src/utils/retrieve_vm_stress.py"
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("Von-Mises stress data extracted successfully.")

        # Check if the target file exists
        source_file = os.path.join(os.getcwd(), "max_vm_stress.txt")
        if os.path.exists(source_file):
            # Ensure the target directory exists
            os.makedirs("src/abaqus_files", exist_ok=True)

            # Move the file to the target directory
            target_file = os.path.join("src/abaqus_files", "max_vm_stress.txt")
            shutil.move(source_file, target_file)
            # print(f"Moved {source_file} to {target_file}")

    except subprocess.CalledProcessError as e:
        print("Error during Abaqus job execution.")
        print("Error output:", e.stderr)
    
    return result.stdout

# def parse_stress_data():
#     parsed_data = []
    
#     # Open and read the file line by line
#     with open('src/abaqus_files/max_vm_stress.txt', 'r') as file:
#         for line in file:
#             line = line.strip()  # Remove any trailing whitespace
            
#             # Check if the line contains data of interest
#             if "Mises stress" in line:
#                 mises_stress = float(line.split("is ")[-1].strip())
#                 # Append the parsed data as a dictionary
#                 parsed_data.append({
#                     "von-Mises Stress": f"{round(mises_stress/10e6, 2)} MPA"
#                 })
    
#     return parsed_data