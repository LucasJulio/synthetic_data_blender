from bpy import ops, context, data
import os

ops.object.select_all(action='SELECT')
ops.object.delete()
objects_folder_path = "/home/ribeiro-desktop/blender_experiments/objects_models/Alembic/"
render_folder_path = "/home/ribeiro-desktop/blender_experiments/render_results/"

for file in os.listdir(objects_folder_path):
	full_path = objects_folder_path + file
	imported_object = ops.wm.alembic_import(filepath=full_path)

# Set camera
context.scene.camera = data.objects['Camera']
context.scene.render.filepath = render_folder_path + "/sucesso.jpg"
ops.render.render(write_still = True)