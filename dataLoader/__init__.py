from .blender import BlenderDataset
from .blender import BlenderMVSDataset
from .dan_video import DanDataset


dataset_dict = {          
                'blender': BlenderDataset,
                }

mvs_dataset_dict = {'blender': BlenderMVSDataset,
                    'dan_data': DanDataset}