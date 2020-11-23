import cv2 as cv
import numpy as np
from glob import glob
import os

DATASETS_MAIN_PATH = os.path.expanduser("/datasets/")
SELECTED_DATASET = "Arduino_2"
ARDUINO_COMPONENTS_CLASSES = {
    # VMAP_ID    # CLASS_ID     # CLASS_NAME
    0: 0,  # "0_Background"

    1: 1,  # "1_Substrate"

    5: 2,  # "2_Header_Female"
    6: 2,  # "2_Header_Female"
    7: 2,  # "2_Header_Female"
    25: 2,  # "2_Header_Female"

    4: 3,  # "3_Header_Male"
    17: 3,  # "3_Header_Male"

    3: 4,  # "4_Main_Microcontroller"

    10: 5,  # "5_USB_Connector"

    32: 6,  # "6_DC_Jack"

    23: 7,  # "7_Button"

    37: 8,  # "8_Voltage_Regulator"
    24: 8,  # "8_Voltage_Regulator"

    9: 9,  # "9_SMD_CDR" # Capacitor_or_Diode_or_Resistor
    12: 9,  # "9_SMD_CDR" # Capacitor_or_Diode_or_Resistor
    15: 9,  # "9_SMD_CDR" # Capacitor_or_Diode_or_Resistor
    18: 9,  # "9_SMD_CDR" # Capacitor_or_Diode_or_Resistor
    19: 9,  # "9_SMD_CDR" # Capacitor_or_Diode_or_Resistor
    20: 9,  # "9_SMD_CDR" # Capacitor_or_Diode_or_Resistor
    21: 9,  # "9_SMD_CDR" # Capacitor_or_Diode_or_Resistor
    26: 9,  # "9_SMD_CDR" # Capacitor_or_Diode_or_Resistor
    28: 9,  # "9_SMD_CDR" # Capacitor_or_Diode_or_Resistor
    29: 9,  # "9_SMD_CDR" # Capacitor_or_Diode_or_Resistor
    30: 9,  # "9_SMD_CDR" # Capacitor_or_Diode_or_Resistor
    48: 9,  # "9_SMD_CDR" # Capacitor_or_Diode_or_Resistor
    51: 9,  # "9_SMD_CDR" # Capacitor_or_Diode_or_Resistor
    53: 9,  # "9_SMD_CDR" # Capacitor_or_Diode_or_Resistor
    55: 9,  # "9_SMD_CDR" # Capacitor_or_Diode_or_Resistor
    57: 9,  # "9_SMD_CDR" # Capacitor_or_Diode_or_Resistor
    59: 9,  # "9_SMD_CDR" # Capacitor_or_Diode_or_Resistor
    61: 9,  # "9_SMD_CDR" # Capacitor_or_Diode_or_Resistor
    63: 9,  # "9_SMD_CDR" # Capacitor_or_Diode_or_Resistor
    73: 9,  # "9_SMD_CDR" # Capacitor_or_Diode_or_Resistor
    75: 9,  # "9_SMD_CDR" # Capacitor_or_Diode_or_Resistor
    77: 9,  # "9_SMD_CDR" # Capacitor_or_Diode_or_Resistor

    81: 9,  # "9_SMD_CDR" # Capacitor_or_Diode_or_Resistor
    83: 9,  # "9_SMD_CDR" # Capacitor_or_Diode_or_Resistor
    85: 9,  # "9_SMD_CDR" # Capacitor_or_Diode_or_Resistor

    13: 10,  # "10_SMD_LED"
    67: 10,  # "10_SMD_LED"
    69: 10,  # "10_SMD_LED"
    71: 10,  # "10_SMD_LED"

    14: 11,  # "11_SO_Transistor"

    27: 12,  # "12_Crystal_Oscillator"

    8: 13,  # "13_Electrolytic_Capacitor"
    43: 13,  # "13_Electrolytic_Capacitor"

    39: 14,  # "14_USB_Controller"

    22: 15,  # "15_IC"

    31: 16,  # "16_Polyfuse"
}

NUMBER_OF_CLASSES = 17


def convert_segmentation_to_mapping(img):
    inter = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    m_img = np.zeros_like(inter).astype(np.uint8)
    val = 0
    step = 255//NUMBER_OF_CLASSES
    low_bound = np.clip(val, 0, 255)
    up_bound = np.clip(low_bound + step, 1, 255)
    for i in range(1, NUMBER_OF_CLASSES + 1):
        m_img[np.logical_and(inter >= low_bound, inter <= up_bound)] = i
        val = val + step
        low_bound = np.clip(val, 1, 255)
        up_bound = np.clip(low_bound + step, 1, 255)

    return m_img


def convert_unique_component_id_to_class(img):
    """
    For the "Arduino" dataset, converts each individual component ID to a particular class, defined
    by a dictionary

    """
    img = np.asarray(img)
    copy = np.copy(img)
    for component_id in ARDUINO_COMPONENTS_CLASSES:
        img[copy == component_id] = ARDUINO_COMPONENTS_CLASSES[component_id]  # Avoids unwanted conversions
    return img


if __name__ == '__main__':
    main_path = os.path.join(DATASETS_MAIN_PATH, SELECTED_DATASET)
    visible_segmentation_imgs_paths = glob(os.path.join(main_path, "visible_maps/*"))
    converted_map_imgs_paths = os.path.join(main_path, "maps/")
    for path in visible_segmentation_imgs_paths:
        vis_map_img = cv.imread(path)
        # map_img = convert_segmentation_to_mapping(vis_map_img)
        map_img = convert_unique_component_id_to_class(vis_map_img)
        map_path = converted_map_imgs_paths + "".join(path.split("/")[-1][1:])  # drops the "v" from prefix
        cv.imwrite(map_path, map_img)
