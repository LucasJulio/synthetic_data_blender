Each dataset has the following structure:

	<DATASET_FOLDER_NAME>
	│
	├── inputs
	│   │
	│   ├── i_0000000.png
	│   │
	│   ├── i_0000001.png
	│   │
	│   └── i_0000002.png
	│
	├── visible_maps
	│   │
	│   ├── vm_0000000.png
	│   │	
	│   ├── vm_0000001.png
	│   │	
	│   └── vm_0000002.png
	│
	├── maps
	│   │
	│   ├── m_0000000.png
	│   │	
	│   ├── m_0000001.png
	│   │	
	│   └── m_0000002.png
	│
	└── labels.csv

	- inputs: 
		Folder containing all input image files. These are the images that will be classified by the models, during and after training. Input image files must be named with the "i_" prefix, followed by a 7-digit identifier.

	- visible_maps: 
		Folder containing all semantic segmentation visible mappings for the corresponding input images.

		These visible mappings are NOT used during model training: they are the result of blender renders, and must first be transformed into simpler mappings (with smaller numerical values). This may be done by the mapping_conversion.py script. Still, the visible mappings are convenient for humans to see (and debug) the semantic segmentations.

		Visible mapping image files must be named with the "vm_" prefix, followed by a 7-digit identifier. Obviously, the identifiers must correspond.

	- maps:

		Folder containing all semantic segmentation mappings for the corresponding input images.

		These mappings are used during model training as the target values. Each pixel of the files assumes an integer value ranging from 1 to N where N is the number of semantic labels.

		Mapping image files must be named with the "m_" prefix, followed by a 7-digit identifier. Obviously, the mapping identifier must corresponding.

	labels.csv: File describing further characteristics of the images (i.e., if there is a defect present or not)