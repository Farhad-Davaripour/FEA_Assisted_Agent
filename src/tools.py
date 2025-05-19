import os
import subprocess
import shutil
import re
import json

def generate_input_file(applied_displacement):
    """
    Generate an Abaqus input file with a specified pipe thickness.

    This function runs an Abaqus script (`create_inp_file.py`) using the provided 
    pipe thickness as an argument, generates the input file, and moves it to the 
    designated directory.

    Args:
        applied_displacement (float): The displacement that the pipe is pushed down which is to be used in the Abaqus model.

    Process:
        1. Constructs the command to execute the Abaqus script with the pipe thickness.
        2. Runs the command using `subprocess.run`.
        3. Moves the generated `.inp` file to the specified destination directory.

    Raises:
        subprocess.CalledProcessError: If the Abaqus command execution fails.

    Side Effects:
        - Moves the generated `cantilever_beam.inp` file to `src/abaqus_files/cantilever_beam.inp`.
        - Prints success or error messages to the console.

    Example:
        generate_input_file(0.015)
    """
    command = f"abaqus cae noGUI=./src/utils/create_inp_file.py -- {applied_displacement}"
    try:
        subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Define the source and destination paths
        source_path = 'cantilever_beam.inp'
        destination_path = 'src/abaqus_files/cantilever_beam.inp'

        # Move the file
        shutil.move(source_path, destination_path)
        # print("File moved successfully.")

    except subprocess.CalledProcessError as e:
        print("Error during Abaqus job execution.")
        print("Error output:", e.stderr)
    
    return f"Input file generated successfully."

def run_abaqus():
    """
    Run the Abaqus job and move output files to the designated directory.

    This function executes an Abaqus job using the input file `cantilever_beam.inp` 
    located in the `src/abaqus_files` directory. Upon successful execution, it moves 
    all generated files starting with `cantilever_beam` from the current working 
    directory to the `src/abaqus_files` directory.

    Process:
        1. Constructs the command to run the Abaqus job with the specified input file.
        2. Executes the command using `subprocess.run`.
        3. Iterates through the current working directory to identify and move all 
           files starting with `cantilever_beam` to the `src/abaqus_files` directory.

    Raises:
        subprocess.CalledProcessError: If the Abaqus job execution fails.

    Side Effects:
        - Moves all Abaqus-generated output files with names starting with 
          `cantilever_beam` to the `src/abaqus_files` directory.
        - Prints success or error messages to the console.

    Example:
        run_abaqus()
    """
    command = f"abaqus job=cantilever_beam input=src/abaqus_files/cantilever_beam.inp"
    try:
        subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
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

    return f"Abaqus job completed successfully."

def extract_von_mises_stress_from_ODB():
    """
    Extract Von-Mises stress data from the Abaqus ODB file.

    This function runs a Python script (`retrieve_vm_stress.py`) using Abaqus to 
    extract Von-Mises stress data. The extracted data is saved in a file named 
    `max_vm_stress.txt`. If the file is generated successfully, it is moved to 
    the `src/abaqus_files` directory.

    Process:
        1. Executes the `retrieve_vm_stress.py` script using the Abaqus Python command.
        2. Checks if the `max_vm_stress.txt` file exists in the current working directory.
        3. Moves the file to the `src/abaqus_files` directory.
        4. Ensures the target directory exists before moving the file.

    Returns:
        str: The standard output from the script execution, containing any logs or results.

    Raises:
        subprocess.CalledProcessError: If the Abaqus Python script execution fails.

    Side Effects:
        - Moves the `max_vm_stress.txt` file to the `src/abaqus_files` directory if it exists.
        - Prints success or error messages to the console.

    Example:
        stress_data = extract_von_mises_stress_from_ODB()
    """
    command = f"abaqus python src/utils/retrieve_vm_stress.py"
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("Von-Mises stress data extracted successfully.")
        
        data = round(float(result.stdout.split("is ")[1].strip())/1e6,2)

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
    
    return f"von_mises stress is {data} MPa"

def parse_stress_mpa(path="src/abaqus_files/max_vm_stress.txt"):
    """Return the stress value in MPa if file exists, else None."""
    try:
        with open(path) as f:
            text = f.read()
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
        if not m:
            return None
        value = float(m.group(1))
        if value > 1000:
            value /= 1e6
        return value
    except OSError:
        return None
    
ACTION_RE = re.compile(r"Action:\s*([A-Za-z0-9_]+)")
INPUT_RE  = re.compile(r"Action Input:\s*(\{.*\})", re.S)

def extract_action(messages):
    """
    Return (tool_name, json_args) from the first assistant message
    that contains an “Action:” line.
    """
    for m in messages:                                 # each element in output_messages
        msg = m.get("message", m)                      # tolerate either shape
        if msg.get("role") != "assistant":
            continue

        content = msg.get("content", "")
        m_action = ACTION_RE.search(content)
        if not m_action:
            continue

        name = m_action.group(1)
        m_args = INPUT_RE.search(content)
        args  = m_args.group(1) if m_args else "{}"

        # round-trip through json so the grader sees valid JSON
        try:
            args = json.dumps(json.loads(args))
        except json.JSONDecodeError:
            args = "{}"

        return name, args

    return None, "{}" 