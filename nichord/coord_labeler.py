from typing import Union, Tuple

import numpy as np
from scipy.spatial import distance

from nichord.peak import _read_atlas_peak

def get_idx_to_label(coords: list, atlas: str = 'yeo',
                     search_closest: bool = True,
                     max_dist: Union[float, int] = 5,
                     must_have=None) -> dict:
    """
    Gets a dictionary where each ROI (idx) in the list of coords is mapped to a
        label. By default, this is mapping to a Yeo network parcellations.
        If the coordinate does not correspond to any parcellation, the string
        "Uncertain" will be used.

    :param coords: List of tuples (x, y, z)
    :param atlas: string. "yeo" is used by default, although you can also
        pass a string corresponding to an atlas from the atlasreader package.
        These include: 'aal', 'aicha', 'desikan_killiany', 'destrieux',
        'harvard_oxford', 'juelich', 'marsatlas', 'neuromorphometrics',
        'talairach_ba', 'talairach_gyrus'.
    :param search_closest: If the coordinate does not perfectly land within
        one of the atlas parcellations, then if search_closest == True,
        the nearest parcellation will be used.
    :param max_dist: If search_closest == True, max_dist corresponds to the
        distance that will be searched before using "Uncertain."
    :param must_have: used to filter what labels are considered valid. see
        find_closest(...)
    :return:
    """

    idx_to_label = {}
    if atlas.lower() == 'yeo':
        atlas = get_yeo_atlas()

    for idx, (x, y, z) in enumerate(coords):
        if search_closest:
            region, dist = find_closest(atlas, [x, y, z], max_dist=max_dist,
                                        must_have=must_have)
        else:
            region = _read_atlas_peak(atlas, [x, y, z])
        region = region.replace('uncertain', 'Uncertain')
        idx_to_label[idx] = region
    return idx_to_label


def find_closest(atlas: Union[str, object], coord: Union[tuple, list],
                 must_have: Union[list, None] = None,
                 max_dist: Union[int, float] = 5) -> Tuple[str, float]:
    """
    This will find the parcellation closest to the passed coord. If no
        parcellation is found within max_dist of the coord, it will be
        labeled 'uncertain'.

    :param atlas: atlas, can either be a string or the Yeo atlas constructed in
        get_idx_to_label(...)
    :param coord: (x, y, z)
    :param must_have: If this is not None, this should be a list of strings.
        The final ROI label must have one of those strings as a substring or
        else it will not be used. For example, if you are interested in getting
        Brodmann areas with the 'talairach_ba' atlas, you would pass
        must_have=["Brodmann"]
    :param max_dist: max_dist corresponds to the distance that will be
        searched before using "Uncertain."
    :return: the label and the distance from the coord to the label
    """

    region = _read_atlas_peak(atlas, coord)
    if isinstance(region, float) and np.isnan(region):
        region = 'uncertain'

    bad_keys = ['uncertain', 'Background', 'Cerebral',
                'Cerebellum']  # these are labels  are too vague and are skipped
    if not any(key in region for key in bad_keys) and (
            must_have is None or any(key in region for key in must_have)):
        return region, 0

    spread_dist = range(-max_dist, max_dist + 1)
    spread_dist = sorted(spread_dist, key=lambda x: abs(x))
    min_dist = max_dist

    min_region = 'uncertain'
    for x_move in spread_dist:
        for y_move in spread_dist:
            for z_move in spread_dist:
                coord_new = [coord[0] + x_move,
                             coord[1] + y_move,
                             coord[2] + z_move]
                dist = distance.euclidean(coord, coord_new)
                if dist > min_dist:
                    continue
                region = _read_atlas_peak(atlas, coord_new)
                if isinstance(region, float) and np.isnan(region):
                    region = 'uncertain'
                if not any(key in region for key in bad_keys) and (
                        must_have is None or any(
                        key in region for key in must_have)):
                    min_region = region
                    min_dist = dist
    return min_region, min_dist


class AttrDict(dict):
    """
    Used in coord_labeler.get_yeo_atlas()
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_yeo_atlas() -> AttrDict:
    """
    Prepares a 7-network version of the Yeo for get_idx_to_label(...)
    :return:
    """

    import pandas as pd # import statement here to speed things up
                        # (not used anywhere else)
    from nilearn import datasets, image
    yeo = AttrDict()
    yeo_fetched = datasets.fetch_atlas_yeo_2011()
    atlas_yeo = yeo_fetched.thick_7
    atlas_yeo = image.load_img(atlas_yeo)
    yeo.image = image.load_img(atlas_yeo)
    yeo.labels = pd.DataFrame({'name': ['uncertain', 'Visual', 'SM', 'DAN',
                                        'VAN', 'Limbic', 'FPCN', 'DMN']})
    yeo.atlas = 'yeo'
    return yeo
