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
        self.polygon_times = np.zeros((1, 0), dtype=float)
        return

    def set(self, frame_id, polygon, time=None):
        if frame_id in self.frame_ids:
            idx = np.argwhere(self.frame_ids == frame_id)
            self.frame_polygons[idx]['polygon'] = polygon
            if time is not None:
                self.frame_polygons[idx]['time'] = time
        else:
            self.frame_ids = np.append(self.frame_ids, np.uint32(frame_id))
            if time is None:
                time = np.nan
            self.polygon_times = np.append(self.polygon_times, float(time))
            self.frame_polygons.append({'frame_id': frame_id, 'time': time, 'polygon': polygon})

        return

    def get_id(self, frame_id):
        """
        get polygon by frame id
        :param frame_id: frame id list / scalar
        :return: polygon points list / scalar (in correspondance with the input)

        """

        if not isinstance(frame_id, list):
            frame_id = [frame_id]

        if len(frame_id) == 1:
            is_scalar = True
        else:
            is_scalar = False

        res_polygons = None
        # get polygon
        if len(self.frame_polygons) > 0:
            # find frame id
            res_polygons = []
            for fid in frame_id:
                self.frame_ids == fid
                idx = self.frame_ids == fid
                if any(idx):
                    res_polygons.append(self.frame_polygons[idx]['polygon'])
                else:
                    res_polygons.append(None)

            if is_scalar:
                res_polygons = res_polygons[0]

        return res_polygons


    def get_time(self, query_times, valid_time_gap=None):
        """
        get polygon by frame time

        :param query_times: query times list / scalar
        :param valid_time_gap: valid time gap to search for polygon relative to query times
                               if None: valid time gap will automatically be calculated from 0.4 polygons time diff median
        :return: polygon points list / scalar (in correspondance with the input)
        """
        if not isinstance(query_times, list):
            query_times = [query_times]

        if len(query_times) == 1:
            is_scalar = True
        else:
            is_scalar = False

        res_polygons = None
        # get polygon
        if len(self.frame_polygons) > 0:

            if valid_time_gap is None:
                time_step =  np.median(np.diff(np.array(self.polygon_times)))
                valid_time_gap = 0.4 * time_step

            # find the closest polygon in time
            res_polygons = []
            for t in query_times:
                idx = np.nanargmin(np.abs(t - self.polygon_times))
                if (not(np.isnan(self.polygon_times[idx])) and
                        np.abs(t - self.polygon_times[idx]) < valid_time_gap):
                    res_polygons.append(self.frame_polygons[idx]['polygon'])
                else:
                    res_polygons.append(None)

            if is_scalar:
                res_polygons = res_polygons[0]

        return res_polygons

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
            self.polygon_times = np.array([x['time'] if 'time' in x.keys() else None for x in self.frame_polygons], dtype=float)

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

