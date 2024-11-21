from odbAccess import openOdb

# Open the ODB file
odb_path = 'src/abaqus_files/cantilever_beam.odb'
odb = openOdb(odb_path)

# Access the stress field values at the last frame of Step-1
stress_field_values = odb.steps['Step-1'].frames[-1].fieldOutputs['S'].values

# List to store Mises stress values
vm_stress_list = [field_value.mises for field_value in stress_field_values]

# Find the maximum Mises stress
max_mises_stress = max(vm_stress_list)

# Print the overall maximum Mises stress
print('Maximum Mises stress in the cantilever beam is %f' % max_mises_stress)

# Write the maximum Mises stress to a text file
with open('max_vm_stress.txt', 'w') as output_file:
    output_file.write('Maximum Mises stress in the cantilever beam is %f\n' % max_mises_stress)

# Close the ODB file
odb.close()
