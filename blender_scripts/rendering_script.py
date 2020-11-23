import colorsys
import os
import random
import re
import string
import numpy as np
from bpy import ops, context, data

# Commented components are children of other components in this list, and this was causing many issues
# TODO: try to get rid of this list
COMPONENTS_TO_HIDE = [
    'AMS1117',
    'ATMEL25',
    'Atmel',
    # 'node5',
    # 'node6.001',
    'Botao',
    'Jumper ICSP',
    # 'node7.001',
    'M7',
    'node8',
    'node9',
    'node10',
    'node13',
    'node13.001',
    # 'Cylinder.001',
    'node18',
    'node20',
    'node20.002',
    'node20.004',
    'node22',
    'node22.001',
    'node22.002',
    'node22.003',
    'node23',
    'node24',
    'node24.002',
    'node24.004',
    'node24.006',
    'node24.008',
    'node24.010',
    'node24.012',
    'node24.014',
    'node24.016',
    'node27',
    'node30',
    'node31',
    'node34',
    'node37',
    'node37.001',
    'node37.002',
    'node37.003',
    'node42',
    'node44',
    'node45',
    'node47',
    'node47.001',
    'node48',
    'node49',
    'node50',
    'node51',
    'node52',
    'node53',
    'Substrato',
    # 'contato',
    # 'contato.001',
    # 'contato.002',
    # 'contato.003',
    # 'contato.004',
    # 'contato.005',
    # 'contato.006',
    # 'contato.007',
    # 'contato.008',
    # 'contato.009',
    # 'contato.010',
    # 'contato.011',
    # 'contato.012',
    # 'contato.013',
    # 'contato.014',
    # 'contato1',
    # 'contato2',
    # 'contatos',
    # 'contatos.001',
    # 'contatos.002',
    # 'Cylinder',
    # 'node17.001',
    # 'node19',
    # 'node20.001',
    # 'node20.003',
    # 'node20.005',
    # 'node24.001',
    # 'node24.003',
    # 'node24.005',
    # 'node24.007',
    # 'node24.009',
    # 'node24.011',
    # 'node24.013',
    # 'node24.015',
    # 'node24.017',
    # 'node25',
    # 'node27.001',
    # 'node43.001'
]

ENVIRONMENT_ROT_DIVISIONS = 60
PROB_SUPP_LIGHT1_ON = 0.3
PROB_SUPP_LIGHT2_ON = 0.3
PROB_SUPP_LIGHT3_ON = 0.3
PROB_SUPP_LIGHT4_ON = 0.3
PROB_SUPP_LIGHT5_ON = 0.3

PROB_HIDE_OBJ = 0.1
PROB_HIDE_TXT = 0.5

MAX_ANGLE_PITCH_CAMERA = 5
MAX_ANGLE_YAW_CAMERA = 2
MAX_DELTA_X_CAMERA = 0.01
MAX_DELTA_Y_CAMERA = 0.01

MAX_PLANE_ROTATION = 360

# root_folder = 'A:\\Documentos\\GitHub\\synthetic_data_blender\\'
root_folder = '/home/ribeiro-desktop/POLI/TCC/blender_experiments'
blend_file = os.path.join(root_folder, 'Kicad files', 'Arduino_Uno', 'arduino_uno_dos.blend')
hdri_folder = os.path.join(root_folder, 'HDRI')
ops.wm.open_mainfile(filepath=blend_file)

# List of possible materials used for the floor plane
floors_list = []
for m in data.materials:
    if 'piso' in m.name:
        floors_list.append(m)


def decision(prob):
    return np.random.rand() < prob


def set_camera_focal_length(fl):
    camera_object = data.objects["Camera"]
    camera_object.select_set(True)
    camera_object.data.lens = fl
    camera_object.select_set(False)


def set_render_exposure(exp):
    context.scene.cycles.film_exposure = exp


def set_evironment_lights(rot_z=False, hdri_filename=None, turn_on_supp1=True, turn_on_supp2=True, turn_on_supp3=True,
                          turn_on_supp4=True, turn_on_supp5=True):
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
    if rot_z:
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

    data.objects["Luz suporte 1"].hide_render = True
    data.objects["Luz suporte 2"].hide_render = True
    data.objects["Luz suporte 3"].hide_render = True
    data.objects["Luz suporte 4"].hide_render = True
    data.objects["Luz suporte 5"].hide_render = True

    if turn_on_supp1 and decision(PROB_SUPP_LIGHT1_ON):
        energy = np.random.randint(4, 12)
        data.objects["Luz suporte 1"].hide_render = False
        data.objects["Luz suporte 1"].data.energy = energy

    if turn_on_supp2 and decision(PROB_SUPP_LIGHT2_ON):
        energy = np.random.randint(4, 12)
        data.objects["Luz suporte 2"].hide_render = False
        data.objects["Luz suporte 2"].data.energy = energy

    if turn_on_supp3 and decision(PROB_SUPP_LIGHT3_ON):
        energy = np.random.randint(4, 12)
        data.objects["Luz suporte 3"].hide_render = False
        data.objects["Luz suporte 3"].data.energy = energy

    if turn_on_supp4 and decision(PROB_SUPP_LIGHT4_ON):
        energy = np.random.randint(4, 12)
        data.objects["Luz suporte 4"].hide_render = False
        data.objects["Luz suporte 4"].data.energy = energy

    if turn_on_supp5 and decision(PROB_SUPP_LIGHT5_ON):
        energy = np.random.randint(4, 12)
        data.objects["Luz suporte 5"].hide_render = False
        data.objects["Luz suporte 5"].data.energy = energy


def hide_objects(probability_to_hide=PROB_HIDE_OBJ):
    """
    Hide some objects at random
    """
    # for obj_name in data.collections['Componentes_Arduino'].all_objects.keys():
    for obj_name in COMPONENTS_TO_HIDE:
        if decision(probability_to_hide):
            data.objects[obj_name].hide_render = True
            hide_children(obj_name)  # Objects' children must be hidden aswell

    data.objects['Substrato'].hide_render = False  # Substract must always be visible!


def hide_children(obj_name):
    obj = data.objects[obj_name]
    if len(obj.children) > 0:
        for c in range(len(obj.children)):
            obj.children[c].hide_render = True
            hide_children(obj.children[c].name)  # Recursion


def hide_text(probability_to_hide=PROB_HIDE_TXT):
    """
    Hide some texts at random
    """
    for txt_name in data.collections['Texto'].all_objects.keys():
        if decision(probability_to_hide):
            data.objects[txt_name].hide_render = True


def show_objects():
    """
    Show all objects in renders of collection Arduino
    """
    for obj_name in data.collections['Arduino'].all_objects.keys():
        data.objects[obj_name].hide_render = False


# TODO: make "undo" option for restoring original position and prevent angles that are bigger than desired
def rotate_camera(max_angle_pitch=None, max_angle_yaw=None):
    handler = data.objects['Handler camera']

    if max_angle_pitch is not None:
        angle_pitch = np.random.uniform(- max_angle_pitch / 2, max_angle_pitch / 2)
        rad_pitch = angle_pitch * np.pi / 180
        handler.rotation_euler[0] = rad_pitch  # X or Pich

    if max_angle_yaw is None:
        angle_yaw = np.random.uniform(- max_angle_yaw / 2, max_angle_yaw / 2)
        rad_yaw = angle_yaw * np.pi / 180
        handler.rotation_euler[2] = rad_yaw  # Z or Yaw


# TODO: make "undo" option for restoring original position and prevent camera drifting
def position_camera(max_delta_x=None, max_delta_y=None):
    handler = data.objects['Handler camera']

    if max_delta_x is not None:
        delta_x = np.random.uniform(-max_delta_x / 2, max_delta_x / 2)
        handler.delta_location[0] = delta_x

    if max_delta_y is not None:
        delta_y = np.random.uniform(-max_delta_y / 2, max_delta_y / 2)
        handler.delta_location[1] = delta_y


def set_plane_material():
    """
    Set plane material randomly from a set of materials with 'piso' in their name
    """
    plane = data.objects['Plane']
    random_material = np.random.randint(0, len(floors_list))
    plane.active_material = floors_list[random_material]


def rotate_plane(max_plane_rotation):
    obj = data.objects['Plane']
    obj.select_set(True)
    rot = np.random.uniform(0, max_plane_rotation * np.pi / 180)
    ops.transform.rotate(value=rot, orient_axis='Z', orient_type='GLOBAL',
                         orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                         orient_matrix_type='GLOBAL', constraint_axis=(False, False, False), mirror=False,
                         use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1,
                         use_proportional_connected=False, use_proportional_projected=False)
    obj.select_set(False)


def set_color_pcb():
    """
    Set PCB and its path a random color
    """

    def srgb_to_linear(r, g, b):
        def srgb(c):
            a = .055
            if c <= .04045:
                return c / 12.92
            else:
                return ((c + a) / (1 + a)) ** 2.4

        return tuple(srgb(c) for c in (r, g, b))

    hue_pcb = np.random.uniform(0.540, 0.617)
    sat_pcb = np.random.uniform(0.804, 1.000)
    val_pcb = np.random.uniform(0.266, 0.371)

    # Converte valores HSV para RGB
    rgb_pcb = colorsys.hsv_to_rgb(hue_pcb, sat_pcb, val_pcb)

    # Converte e sRGB para Linear
    rgb_pcb = srgb_to_linear(rgb_pcb[0], rgb_pcb[1], rgb_pcb[2])

    # Converte valores HSV para RGB
    rgb_trilha = colorsys.hsv_to_rgb(hue_pcb + 0.015, sat_pcb, val_pcb - 0.094)

    # Converte e sRGB para Linear
    rgb_trilha = srgb_to_linear(rgb_trilha[0], rgb_trilha[1], rgb_trilha[2])

    data.materials["Pcb_a3ee"].node_tree.nodes["ColorRamp"].color_ramp.elements[1].color = rgb_pcb + (1,)
    data.materials["Pcb_a3ee"].node_tree.nodes["ColorRamp"].color_ramp.elements[0].color = rgb_trilha + (1,)


class ChangeText:
    """
    Class responsible to assining random text in objects within "Texto" collection and begining with "Text" in their names
    """

    def __init__(self):
        # Pega o estado dos textos antes
        self.text_before = []
        for obj in data.collections['Texto'].all_objects:
            if 'Text' in obj.name:
                self.text_before.append(obj.data.body)

    @staticmethod
    def create_random_text(N):
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))

    @staticmethod
    def separate(s):
        return re.split('([ |\n|-])', s)

    def change_text(self):
        """
        Write random text
        """
        for obj in data.collections['Texto'].all_objects:
            if 'Text' in obj.name:
                str_list = self.separate(obj.data.body)
                for i, word in enumerate(str_list):
                    if word not in [' ', '\n', '-']:
                        N = len(word)
                        str_list[i] = self.create_random_text(N)

                # Concatenacao da string randomica
                random_string = ''.join(str_list)

                # Escrevendo string
                obj.data.body = random_string

    def back_to_original_text(self):
        """
        Write the original text
        """
        for i, obj in enumerate(data.collections['Texto'].all_objects):
            if 'Text' in obj.name:
                obj.data.body = self.text_before.pop(0)


# Set camera
context.scene.camera = data.objects["Camera"]
show_objects()

# Create the text randomizer
# random_text = ChangeText()  # TODO: Not working; Use when fixed

for i in range(440, 2000):
    img_id = str(i).zfill(7)

    r = np.random.random(6)
    r = r - 0.5

    set_evironment_lights(rot_z=False)

    if decision(0.5):
        hide_objects(probability_to_hide=PROB_HIDE_OBJ)

    if decision(0.5):
        hide_text(probability_to_hide=PROB_HIDE_TXT)

    focal_length = np.random.randint(30, 40)
    set_camera_focal_length(focal_length)

    exposure = np.random.random() * 2.0 + 0.8
    set_render_exposure(exposure)
    rotate_plane(max_plane_rotation=MAX_PLANE_ROTATION)
    # rotate_camera(max_angle_pitch=MAX_ANGLE_PITCH_CAMERA, max_angle_yaw=MAX_ANGLE_YAW_CAMERA)
    # position_camera(max_delta_x=MAX_DELTA_X_CAMERA, max_delta_y=MAX_DELTA_Y_CAMERA)
    set_plane_material()

    # Assing a color to the PCB, close to the original color in the HSV space
    set_color_pcb()

    # Assing random text to the text objects
    # random_text.change_text()  # TODO: Use when fixed

    # Nodes and renders
    nodes = data.scenes[0].node_tree.nodes
    file_output_node = nodes["File Output"]
    file_output_node.file_slots[0].path = "vm_" + img_id + '#'  # Visible Maps
    file_output_node = nodes["File Output.001"]
    file_output_node.file_slots[0].path = "i_" + img_id + '#'  # Inputs
    ops.render.render(write_still=True, use_viewport=True)

    show_objects()

    # Return text back to the original
    # random_text.back_to_original_text()  # TODO: Use when fixed
