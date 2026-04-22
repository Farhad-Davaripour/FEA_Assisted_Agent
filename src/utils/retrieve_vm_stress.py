import sys
from odbAccess import openOdb


def _resolve_odb_path():
    # Abaqus forwards args as: script.py -- <arg1> ...
    args = [arg for arg in sys.argv[1:] if arg != "--"]
    if args:
        return args[0]
    return "src/abaqus_files/cantilever_beam.odb"


odb_path = _resolve_odb_path()
odb = openOdb(odb_path)

step_names = list(odb.steps.keys())
if not step_names:
    raise RuntimeError("No analysis steps found in ODB file.")

last_step_name = step_names[-1]
stress_field_values = odb.steps[last_step_name].frames[-1].fieldOutputs["S"].values
vm_stress_list = [field_value.mises for field_value in stress_field_values]
max_mises_stress = max(vm_stress_list)

print("Maximum Mises stress in the model is %f" % max_mises_stress)

with open("max_vm_stress.txt", "w") as output_file:
    output_file.write("Maximum Mises stress in the model is %f\n" % max_mises_stress)

odb.close()
