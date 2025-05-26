import os
import numpy as np
import yaml


def _closest_below(arr, target):
    """
    find index of closest element from below in np.array with reference to target value
    assuming array is sorted

    param: arr - 1D numpy array
    param: target - target value
    return: idx - index of closest element from below in np.array with reference to target value
    """
    # Filter the array to keep only elements <= target
    filtered_arr = arr[arr <= target]

    # If the filtered array is not empty, return the maximum element
    if filtered_arr.size > 0:
        v = np.max(filtered_arr)
        idx = np.argwhere(arr == v).flatten()[0]
    else:
        idx =  None  # If no elements are <= target

    return idx

def _closest_above(arr, target):
    """
    find index of closest element from above in np.array with reference to target value
    assuming array is sorted

    param: arr - 1D numpy array
    param: target - target value
    return: idx - index of closest element from above in np.array with reference to target value
    """

    # Filter the array to keep only elements <= target
    filtered_arr = arr[arr >= target]

    # If the filtered array is not empty, return the maximum element
    if filtered_arr.size > 0:
        v = np.min(filtered_arr)
        idx = np.argwhere(arr == v).flatten()[0]
    else:
        idx = None  # If no elements are <= target

    return idx

class PolygonPerFrame:
    """
    This class handles a record of frame ids and corresponding polygons
    """
    def __init__(self, frame_size):
        self.frame_size = frame_size
        self.frame_polygons = []
        self.frame_ids = np.zeros((1, 0), dtype=np.uint32)
        return

    def set(self, frame_id, polygon):
        if frame_id in self.frame_ids:
            idx = np.argwhere(self.frame_ids == frame_id)
            self.frame_polygons[idx]['polygon'] = polygon
        else:
            self.frame_ids = np.append(self.frame_ids, np.uint32(frame_id))
            self.frame_polygons.append({'frame_id': frame_id, 'polygon': polygon})
        return

    def get(self, frame_id):
        idx = np.argwhere(self.frame_ids == frame_id).flatten()
        if idx.size == 0:
            polygon = None
        else:
            idx = idx[0]
            polygon = self.frame_polygons[idx]['polygon']
        return polygon

    def save(self, file_path):
        folder_name = os.path.dirname(file_path)
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        with open(file_path, 'w') as outfile:
            data = {'frame_size': [self.frame_size[0], self.frame_size[1]], 'frame_polygons': self.frame_polygons}
            yaml.dump(data, outfile, default_flow_style=None, sort_keys=False)
        return


    def load(self, file_path):
        if not os.path.isfile(file_path):
            raise Exception('file: {} not found!'.format(file_path))

        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            self.frame_size = data['frame_size']
            self.frame_polygons = data['frame_polygons']
            self.frame_ids = np.array([x['frame_id'] for x in self.frame_polygons], dtype=np.uint32)

        return

    def interpolate_polygons(self, frame_ids, verbose=False):
        """
        linear interpolation of polygons for all frame ids given reference for some frame ids.
        for this all polygons must be the same number of points!

        param: frame_ids - list of frame ids
        """

        # check if all polygons are the same size!
        n = np.array(self.frame_polygons[0]['polygon']).size / 2
        for p in self.frame_polygons:
            m = np.array(p['polygon']).size / 2
            if m != n:
                print('Warning! not all polygons are the same size! skipping interpolate_polygons')
                return

        res_polygons = []

        # pad to beginning
        if min(self.frame_ids) > min(frame_ids):
            self.set(min(frame_ids), self.frame_polygons[0]['polygon'])
        # pad to end
        if max(self.frame_ids) < max(frame_ids):
            self.set(max(frame_ids), self.frame_polygons[-1]['polygon'])

        # add new frame ids
        for fid in frame_ids:

            if fid in self.frame_ids:
                if verbose:
                    print('frame {} reference polygon polygon:'.format(fid))
                    print(current_polygon)

            else:
                # find prev reference
                idx1 = _closest_below(self.frame_ids, fid)
                prev_poly = np.array(self.frame_polygons[idx1]['polygon'])
                prev_t = self.frame_polygons[idx1]['frame_id']

                # find next reference
                idx2 = _closest_above(self.frame_ids, fid)
                next_poly = np.array(self.frame_polygons[idx2]['polygon'])
                next_t = self.frame_polygons[idx2]['frame_id']

                # interpolate
                a = (next_t - fid) / (next_t - prev_t)
                current_polygon = prev_poly * a + next_poly * (1 - a)

                self.set(fid, current_polygon.tolist())

                if verbose:
                    print('frame {} a={} interpolated polygon:'.format(fid, a))
                    print(current_polygon)

        # sort by frame id
        self.frame_polygons = sorted(self.frame_polygons, key=lambda x: x['frame_id'])
        self.frame_ids = np.sort(self.frame_ids)
        return res_polygons

