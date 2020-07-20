from bpy import ops, context, data
import numpy as np


# UTILS
def move_light(x, y, z, undo_random=False):
    if undo_random:
        x = -x
        y = -y
        z = -z

    light_object = data.objects["Luz_direcional"]
    light_object.select_set(True)
    context.view_layer.objects.active = light_object
    ops.transform.translate(value=(x, y, z), orient_type='GLOBAL',
                            orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True,
                            use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1,
                            use_proportional_connected=False, use_proportional_projected=False)

    light_object.select_set(False)


def set_camera_focal_length(fl):
    camera_object = data.objects["Camera"]
    camera_object.select_set(True)
    camera_object.data.lens = fl
    camera_object.select_set(False)


def set_render_exposure(exp):
    context.scene.cycles.film_exposure = exp


def change_ooi_position(x, y, rot, undo_random=False):
    """
    Changes object of interest positions
    """

    if undo_random:
        rot = -rot
        x = -x
        y = -y

    obj = data.objects["Substrato"]  # TODO: Inconveniently requires that all other parts are "locked" to substrate. Fix
    obj.select_set(True)


    ops.transform.rotate(value=rot, orient_axis='Z', orient_type='GLOBAL',
                         orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                         orient_matrix_type='GLOBAL', constraint_axis=(False, False, False), mirror=False,
                         use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1,
                         use_proportional_connected=False, use_proportional_projected=False)

    # TODO: fix this and make it work once again
    '''
    ops.transform.translate(value=(x, y, 0), orient_type='GLOBAL',
                            orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                            orient_matrix_type='GLOBAL', constraint_axis=(True, True, False), mirror=True,
                            use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1,
                            use_proportional_connected=False, use_proportional_projected=False)
    '''

    obj.select_set(False)


blend_file = "translucido.blend"
blend_folder = "/home/ribeiro-desktop/blender_experiments/blend_files/"
render_folder = "/home/ribeiro-desktop/blender_experiments/render_results/"
ops.wm.open_mainfile(filepath=blend_folder + blend_file)

# Set camera
context.scene.camera = data.objects["Camera"]


for i in range(0, 2001):
    img_id = str(i).zfill(7)
    context.scene.render.filepath = render_folder + "vm_" + img_id + ".png"  # Visible Maps
    r = np.random.random(6)
    r = r - 0.5

    x_light, y_light, z_light = np.random.random()*15, np.random.random()*15, np.random.random()*10
    move_light(x_light, y_light, z_light)

    x_ooi, y_ooi, z_ooi = np.random.random() * 0.1, np.random.random() * 0.1, np.random.random() * 7
    change_ooi_position(x_ooi, y_ooi, z_ooi)

    focal_length = np.random.randint(35, 80)
    set_camera_focal_length(focal_length)

    exposure = np.random.random() * 5.4 + 0.6
    set_render_exposure(exposure)

    # Nodes and renders
    nodes = data.scenes[0].node_tree.nodes
    file_output_node = nodes["File Output"]
    file_output_node.file_slots[0].path = "i_" + img_id + ".png"  # Inputs
    ops.render.render(write_still=True, use_viewport=True)

    # Move everything back to where they were
    move_light(r[0], r[1], r[2], undo_random=True)
    change_ooi_position(r[3], r[4], r[5], undo_random=True)
