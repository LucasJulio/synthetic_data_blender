from bpy import ops, context, data
import os
from numpy.random import random

blend_file = "ORIGINAL_plus_LABELS.blend"
blend_folder = "/home/ribeiro-desktop/blender_experiments/blend_files/"
render_folder = "/home/ribeiro-desktop/blender_experiments/render_results/" 

ops.wm.open_mainfile(filepath=blend_folder + blend_file)

# Set camera
context.scene.camera = data.objects["Camera"]

for i in range(10):
	context.scene.render.filepath = render_folder + str(i) + "_labeled_"
	r = random(10)
	r = r - 0.5
	# Change light position
	light_object = data.objects["Luz"]
	light_object.select_set(True)
	context.view_layer.objects.active = light_object
	ops.transform.translate(value=(r[0]*15, r[1]*15, r[2]*10), orient_type='GLOBAL', orient_matrix=((r[3], r[4], r[5]), (r[6], r[7], r[8]), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
	light_object.select_set(False)

	# Rotate PCB
	pcb_collection = data.collections.get("Placa")
	for obj in pcb_collection.objects:
		obj.select_set(True)

	# ops.transform.translate(value=(r[0]*15, r[1]*15, 0), orient_type='GLOBAL', orient_matrix=((r[3], r[4], 0), (r[6], r[7], 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
	ops.transform.rotate(value=r[9], orient_axis='Z', orient_type='GIMBAL', orient_matrix=((1, 0, -0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GIMBAL', constraint_axis=(False, False, True), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)

	pcb_collection = data.collections.get("Placa")
	for obj in pcb_collection.objects:
		obj.select_set(False)

	nodes = data.scenes[0].node_tree.nodes 
	file_output_node = nodes["File Output"] 
	file_output_node.file_slots[0].path = str(i) + "_original"
	ops.render.render(write_still=True, use_viewport=True)
