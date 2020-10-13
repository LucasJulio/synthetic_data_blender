from bpy import ops, context, data, types
import numpy as np
import os

ENVIRONMENT_ROT_DIVISIONS = 60
PROB_SUPP_LIGHT1_ON = 0.3
PROB_SUPP_LIGHT2_ON = 0.3

PROB_HIDE_OBJ = 0.1
PROB_HIDE_TEXT = 0.6

MAX_ANGLE_PITCH_CAMERA = 26
MAX_DELTA_X_CAMERA = 0.01
MAX_DELTA_Y_CAMERA = 0.01


# render_folder = "/home/ribeiro-desktop/blender_experiments/render_results/"
# root_folder = 'A:\\Documentos\\GitHub\\synthetic_data_blender\\'
root_folder = '/home/ribeiro-desktop/blender_experiments'
blend_file = os.path.join(root_folder, 'Kicad files', 'Arduino_Uno', 'arduino_uno.blend')
hdri_folder = os.path.join(root_folder, 'HDRI')
# blend_folder = "/home/ribeiro-desktop/blender_experiments/blend_files/"
# render_folder = "/home/ribeiro-desktop/blender_experiments/render_results/"
ops.wm.open_mainfile(filepath = blend_file)

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

    obj.select_set(False)


def set_evironment_lights(rot_z=None, hdri_filename=None, turn_on_supp1=None, turn_on_supp2=None):
    """
    Set random environment lights via HDRI image, it's rotation and 2 support lights for more reflexity artifacts
    """
    # Get random enviroment image
    if hdri_filename is None:
        list_hdri_images = [x for x in os.listdir(hdri_folder)]
        index = np.random.randint(low=0, high=len(list_hdri_images))
        hdri_filename = list_hdri_images[index]

    filepath = os.path.join(hdri_folder, hdri_filename)

    hdri_image = data.images.load(filepath=filepath)

    # Set enviroment image
    data.worlds["World"].node_tree.nodes["Environment Texture"].image = hdri_image

    # Calculate random rotation of the environment
    if rot_z is None:
        divisions = ENVIRONMENT_ROT_DIVISIONS
        step = 2 * np.pi / divisions
        # Make a list of possible rotations
        possible_rot = np.zeros(divisions)
        for i in range(0, divisions):
            possible_rot[i] = step * i

        index = np.random.randint(low=0, high=divisions)
        rot_z = possible_rot[index]

    # Set enviroment rotation
    data.worlds["World"].node_tree.nodes["Mapping"].inputs[2].default_value[2] = rot_z

    # Set support lights
    # Probability to turn on
    def decision(prob):
        return np.random.rand() < prob

    if turn_on_supp1 is None:
        turn_on_supp1 = decision(PROB_SUPP_LIGHT1_ON)
    if turn_on_supp2 is None:
        turn_on_supp2 = decision(PROB_SUPP_LIGHT2_ON)

    # Inverse of turn_on_supp because of hide instead of unhide
    data.objects["Luz suporte 1"].hide_render = np.logical_not(turn_on_supp1)
    data.objects["Luz suporte 2"].hide_render = np.logical_not(turn_on_supp2)


def hide_objects():
    """
    Hide some objects at random based on PROB_HIDE_OBJ and PROB_HIDE_TEXT
    """

    def decision(prob):
        return np.random.rand() < prob

    def pass_hide_children(obj):
        hide = obj.hide_render
        for child in obj.children:
            child.hide_render = hide
            pass_hide_children(child)

    for obj in data.collections['Arduino'].all_objects:
        # print(obj.name)
        if (obj is not None) and isinstance(obj, types.Object):
            if obj.users_collection[0].name != 'Texto':
                obj.hide_render = decision(PROB_HIDE_OBJ)

            # Increased Probability of hiding text
            elif obj.users_collection[0].name == 'Texto':
                obj.hide_render = decision(PROB_HIDE_TEXT)

    # Pass decision to objects children
    for obj in data.collections['Arduino'].all_objects:

        # Check if obj has children
        if (obj is not None) and isinstance(obj, types.Object):
            if len(obj.children) > 0:
                # Pass decision to children of obj
                pass_hide_children(obj)

    # Unhide Substrato
    data.objects['Substrato'].hide_render = False


def unhide_objects():
    """
    Unhide all objects of collection Arduino
    """
    for obj in data.collections['Arduino'].all_objects:
        # print(type(obj))
        if (obj is not None) and isinstance(obj, types.Object):
            # print(type(obj))
            obj.hide_render = False


def rotate_camera(angle_pitch=None, angle_yaw=None):
    handler = data.objects['Handler camera']

    if angle_pitch is None:
        angle_pitch = np.random.random() * MAX_ANGLE_PITCH_CAMERA

    if angle_yaw is None:
        angle_yaw = np.random.random() * 360

    # Convert to radians
    rad_pitch = angle_pitch * np.pi / 180
    rad_yaw = angle_yaw * np.pi / 180

    handler.rotation_euler[0] = rad_pitch  # X or Pich
    handler.rotation_euler[2] = rad_yaw  # Z or Yaw


def position_camera(delta_x=None, delta_y=None):
    handler = data.objects['Handler camera']

    if delta_x is None:
        delta_x = (np.random.random() * 2 * MAX_DELTA_X_CAMERA) - MAX_DELTA_X_CAMERA

    if delta_y is None:
        delta_y = (np.random.random() * 2 * MAX_DELTA_Y_CAMERA) - MAX_DELTA_Y_CAMERA

    handler.delta_location[0] = delta_x  # X delta position
    handler.delta_location[1] = delta_y  # Y delta position


def set_plane_material():
    """
    Set plane material randomly from a set of materials with 'piso' in their name
    """
    plane = data.objects['Plane']

    list_materials = []
    for m in data.materials:
        if 'piso' in m.name:
            list_materials.append(m)

    i = np.random.randint(0, len(list_materials))

    plane.active_material = list_materials[i]


# Set camera
context.scene.camera = data.objects["Camera"]
# unhide_objects()
# first_image=209
for i in range(4000, 5000):
    img_id = str(i).zfill(7)
    # context.scene.render.filepath = render_folder + "vm_" + img_id + ".png"  # Visible Maps
    r = np.random.random(6)
    r = r - 0.5

    # x_light, y_light, z_light = np.random.random()*15, np.random.random()*15, np.random.random()*10
    # move_light(x_light, y_light, z_light)
    set_evironment_lights()

    # x_ooi, y_ooi, z_ooi = np.random.random() * 0.1, np.random.random() * 0.1, np.random.random() * 7
    # change_ooi_position(x_ooi, y_ooi, z_ooi)

    # hide_objects()

    focal_length = np.random.randint(35, 80)
    set_camera_focal_length(focal_length)

    exposure = np.random.random() * 4.0 + 0.6
    set_render_exposure(exposure)

    rotate_camera()
    position_camera()
    set_plane_material()

    # Nodes and renders
    nodes = data.scenes[0].node_tree.nodes
    file_output_node = nodes["File Output"]
    file_output_node.file_slots[0].path = "vm_" + img_id + '#'  # Visible Maps
    file_output_node = nodes["File Output.001"]
    file_output_node.file_slots[0].path = "i_" + img_id + '#'  # Inputs
    ops.render.render(write_still=True, use_viewport=True)

    # Hide support lights
    data.objects["Luz suporte 1"].hide_render = True
    data.objects["Luz suporte 2"].hide_render = True

    # unhide_objects()

    # Move everything back to where they were
    # move_light(r[0], r[1], r[2], undo_random=True)
    # change_ooi_position(r[3], r[4], r[5], undo_random=True)
