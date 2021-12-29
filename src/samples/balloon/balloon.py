import numpy as np

from preprocess import preprocess

BALLOON_CONFIG = {'class_dict': {'balloon': 1, 'background': 0}, 'num_classes': 2, }
BALLOON_CONFIG.update({'meta_shape': 1 + 3 + 3 + 4 + 1 + BALLOON_CONFIG['num_classes']})


class BalloonDataset(preprocess.SegmentationDataset):

    def get_points_from_annotation(self, annotation_key):
        """
         Get polygon points for a segment. [[x1,y1], [x2, y2], ....[]]
        Args:
            annotation_key: key to get info about polygons to make masks

        Returns: polygon_data_list, class_id_list
        """

        polygon_data_list = []
        class_id_list = []
        _region_dict = self.annotation_dict[annotation_key]['regions']
        # If there is more than one object described as polygons, find each class id for each polygon
        # If there is no information about classed in 'region_attributes', add class 1 as binary

        for region_key in _region_dict.keys():

            region = _region_dict[region_key]

            if 'all_points_x' not in region['shape_attributes'].keys():
                print(f'\n[SegmentationDataset] Skipping incorrect observation:\n',
                      f"""annotation_key: {annotation_key}\n_region_list: {region['shape_attributes']}\n""")
                continue
            polygon_points = [[x, y] for x, y in zip(region['shape_attributes']['all_points_x'],
                                                     region['shape_attributes']['all_points_y'])]
            polygon_data_list.append(np.array([polygon_points]))

            # If there is no any keyfields for classes, mark everything as class 1
            if len(region['region_attributes'].keys()) == 0:
                class_id_list.append(1)
            else:
                # In VGG Image Annotator there is an option to add attributes for polygons.
                # We can write class_name to the specified attribute of a polygon
                # For example, by default, attribute name which contains class name is 'object'
                class_name = region['region_attributes'][self.class_key]
                if len(class_name) == 0:
                    raise ValueError(f'Class name is empty. Full annotation: {region}')
                class_id_list.append(self.classes_dict[class_name])

        return polygon_data_list, class_id_list
