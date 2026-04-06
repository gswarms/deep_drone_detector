import copy
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
        polygon = self._to_list(polygon)

        if frame_id in self.frame_ids:
            idx = int(np.argwhere(self.frame_ids == frame_id).flatten()[0])
            self.frame_polygons[idx]['polygon'] = polygon
            if time is not None:
                self.frame_polygons[idx]['time'] = float(time)
        else:
            self.frame_ids = np.append(self.frame_ids, np.uint32(frame_id))
            if time is None:
                time = np.nan
            self.polygon_times = np.append(self.polygon_times, float(time))
            self.frame_polygons.append({'frame_id': int(frame_id), 'time': float(time), 'polygon': polygon})

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
                if fid in self.frame_ids:
                    idx = int(np.argwhere(self.frame_ids == fid).flatten()[0])
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
                # d1 = np.diff(np.array(self.polygon_times))
                # d2 = np.diff(np.array(self.frame_ids))
                # time_step =  np.median(np.divide(d1,d2))
                # valid_time_gap = 0.4 * time_step
                valid_time_gap = np.inf

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
            self.frame_ids = np.array([int(x['frame_id']) for x in self.frame_polygons], dtype=np.uint32)
            self.polygon_times = np.array([float(x['time']) if 'time' in x.keys() else None for x in self.frame_polygons], dtype=float)

        return

    def interpolate_polygons(self, frame_ids, frame_times=None, verbose=False):
        """
        add polygons for new frames by interpolation from existing frames

        Notes:
        1. extrapolation to new frames before the first existing frame, or after the last frame
           will be done by keeping first / last polygon
        2. if frame_times is supplied, and all existing frames have time, interpolatino will be done by time
           if not, interpolation will be done by frame ids
        3. polygons for all existing frames must be os the same number of points

        param: frame_ids - list of new frame ids
        param: frame_times - list of corresponding frame times
        param: verbose - bool
        """
        # check inputs
        if frame_times is not None and len(frame_ids) != len(frame_times):
            raise Exception('invalid input size!')

        frame_ids = np.array(frame_ids).flatten()
        if frame_times is not None:
            frame_times = np.array(frame_times).flatten()
            idx = np.argsort(frame_times)
            frame_times = frame_times[idx]
            frame_ids = frame_ids[idx]
        else:
            frame_ids = np.sort(frame_ids)

        # check if all polygons are the same size!
        n = np.array(self.frame_polygons[0]['polygon']).size / 2
        for p in self.frame_polygons:
            m = np.array(p['polygon']).size / 2
            if m != n:
                print('Warning! not all polygons are the same size! skipping interpolate_polygons')
                return

        # check if we can do time based interpolation
        if (frame_times is not None) and not(np.isnan(self.polygon_times).any()):
            interp_by_time = True
        else:
            interp_by_time = False

        res_polygons = []

        # pad to beginning
        if interp_by_time:
            if min(self.polygon_times) > frame_times[0]:
                p = self.get_time(min(self.polygon_times))
                pc = copy.deepcopy(p)
                self.set(frame_ids[0], pc, frame_times[0])
        else:
            if min(self.frame_ids) > frame_ids[0]:
                p = self.get_id(min(self.frame_ids))
                pc = copy.deepcopy(p)
                if frame_times is not None:
                    idx = self.frame_ids == min(self.frame_ids)
                    t = self.polygon_times[idx]
                    tc = copy.deepcopy(t)
                    self.set(frame_ids[0], pc, tc)
                else:
                    self.set(frame_ids[0], pc)

        # pad to end
        if interp_by_time:
            if max(self.polygon_times) < frame_times[-1]:
                p = self.get_time(max(self.polygon_times))
                pc = copy.deepcopy(p)
                self.set(frame_ids[-1], pc, frame_times[-1])
        else:
            if max(self.frame_ids) < frame_ids[-1]:
                p = self.get_id(max(self.frame_ids))
                pc = copy.deepcopy(p)
                if frame_times is not None:
                    idx = self.frame_ids == max(self.frame_ids)
                    t = self.polygon_times[idx]
                    tc = copy.deepcopy(t)
                    self.set(frame_ids[-1], pc, tc)
                else:
                    self.set(frame_ids[-1], pc)

        # add new frame ids
        for i, fid in enumerate(frame_ids):
            if fid in self.frame_ids:
                if verbose:
                    print('frame {} reference polygon polygon:'.format(fid))
                    current_polygon = self.get_id(fid)
                    print(current_polygon)
            else:
                if interp_by_time:
                    est_time = frame_times[i]
                    # find prev reference
                    idx1 = _closest_below(self.polygon_times, est_time)
                    prev_time = self.polygon_times[idx1]
                    prev_poly = np.array(self.get_time(prev_time))
                    # find next reference
                    idx2 = _closest_above(self.polygon_times, est_time)
                    next_time = self.polygon_times[idx2]
                    next_poly = np.array(self.get_time(next_time))

                else:
                    est_time = fid
                    # find prev reference
                    idx1 = _closest_below(self.frame_ids, fid)
                    prev_poly = np.array(self.frame_polygons[idx1]['polygon'])
                    prev_time = self.frame_polygons[idx1]['frame_id']
                    # find next reference
                    idx2 = _closest_above(self.frame_ids, fid)
                    next_poly = np.array(self.frame_polygons[idx2]['polygon'])
                    next_time = self.frame_polygons[idx2]['frame_id']

                # interpolate
                a = (next_time - est_time) / (next_time - prev_time)
                est_polygon = prev_poly * a + next_poly * (1 - a)
                est_polygon = np.round(est_polygon).astype(int)

                if frame_times is not None:
                    self.set(fid, est_polygon.tolist(), frame_times[i])
                else:
                    self.set(fid, est_polygon.tolist())

                if verbose:
                    print('frame {} a={} interpolated polygon:'.format(fid, a))
                    print(est_polygon)

        # sort
        self.frame_polygons = sorted(self.frame_polygons, key=lambda x: x['frame_id'])
        self.frame_ids = np.sort(self.frame_ids)
        return res_polygons

    @staticmethod
    def _to_list(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        elif isinstance(x, (list, tuple)):
            return list(x)
        else:
            return [x]  # optional: wrap scalar into list