'''
This code was simply taken from atlasreader package and modified slightly
   to no longer use nibabel. In testing the NiChord package, it seemed that
   the nibabel dependency (via atlasreader) was creating issues?
I'm not quite sure but this works per my tests.
Here is a link to the atlasreader GitHub: https://github.com/miykael/atlasreader
'''

import os.path as op
from pkg_resources import resource_filename
from nilearn import image
import numpy as np
import pandas as pd
from sklearn.utils import Bunch


# check_atlas(atlastype) is fairly slow, and so this global dictionary holds
#   onto its output, so it does not need to be done more than once for each
#   `atlastype`
TIME_SAVER = {}

_ATLASES = [
    'aal',
    'aicha',
    'desikan_killiany',
    'destrieux',
    'harvard_oxford',
    'juelich',
    'marsatlas',
    'neuromorphometrics',
    'talairach_ba',
    'talairach_gyrus',
]

_DEFAULT = [
    'aal',
    'desikan_killiany',
    'harvard_oxford',
]

def read_atlas_peak(atlastype, coordinate, prob_thresh=5):
    """
    Returns label of `coordinate` from corresponding `atlastype`

    If `atlastype` is probabilistic, `prob_thresh` determines (in percentage
    units) the threshold to apply before getting label of `coordinate`

    Parameters
    ----------
    atlastype : str
        Name of atlas to use
    coordinate : list of float
        x, y, z MNI coordinates of voxel
    prob_thresh : [0, 100] int, optional
        Probability (percentage) threshold to apply if `atlastype` is
        probabilistic

    Returns
    ------
    label : str or list of lists
        If `atlastype` is deterministic, this is the corresponding atlas label
        of `coordinate`. If `atlastype` is probabilistic, this is a list of
        lists where each entry denotes the probability and corresponding label
        of `coordinate`.
    """

    # get atlas data

    if isinstance(atlastype, str) and tuple(atlastype) not in TIME_SAVER:
        atlastype_copy = tuple(atlastype)
        checked_atlastype = check_atlases(atlastype)
        if type(checked_atlastype) == list:
            if not len(checked_atlastype) == 1:
                raise ValueError(
                    '\'{}\' is not a string or a single atlas. \'all\' '
                    'and \'default\' or not valid inputs.'.format(atlastype))
            else:
                atlastype = checked_atlastype[0]
        TIME_SAVER[atlastype_copy] = atlastype
    elif isinstance(atlastype, str):
        atlastype = TIME_SAVER[tuple(atlastype)]

    try:
        data = atlastype.image.get_fdata()
        # get voxel index
    except:
        data = atlastype.image

    voxID = coord_xyz_to_ijk(atlastype.image.affine, coordinate).squeeze()
    voxID = check_atlas_bounding_box(voxID, data.shape)
    # get label information
    # probabilistic atlas is requested
    if atlastype.atlas.lower() in ['juelich', 'harvard_oxford']:
        probs = data[voxID[0], voxID[1], voxID[2]]
        probs[probs < prob_thresh] = 0
        idx, = np.where(probs)

        # if no labels found
        if len(idx) == 0:
            return [[0, 'no_label']]

        # sort list by probability
        idx = idx[np.argsort(probs[idx])][::-1]

        # get probability and label names
        probLabel = [[probs[i], get_label(atlastype, i)] for i in idx]

        return probLabel
    # non-probabilistic atlas is requested
    else:
        labelID = int(data[voxID[0], voxID[1], voxID[2]])
        label = get_label(atlastype, labelID)
        return label

def check_atlases(atlases):
    """
    Checks atlases

    Parameters
    ----------
    atlases : str or list
        Name of atlas(es) to use

    Returns
    -------
    atlases : list
        Names of atlas(es) to use
    """
    if isinstance(atlases, str):
        atlases = [atlases]
    elif isinstance(atlases, dict):
        if all(hasattr(atlases, i) for i in ['image', 'atlas', 'labels']):
            return atlases
    if 'all' in atlases:
        draw = _ATLASES
    elif 'default' in atlases:
        draw = _DEFAULT
    else:
        draw = atlases

    return [get_atlas(a) if isinstance(a, str) else a for a in draw]


def coord_xyz_to_ijk(affine, coords):
    """
    Converts voxel `coords` in `affine` space to cartesian space

    Parameters
    ----------
    affine : (4, 4) array-like
        Affine matrix
    coords : (N,) list of list
        Image coordinate values, where each entry is a length three list of int
        denoting xyz coordinates in `affine` space

    Returns
    ------
    ijk : (N, 3) numpy.ndarray
        Provided `coords` in cartesian space
    """
    coords = _check_coord_inputs(coords)
    vox_coords = np.linalg.solve(affine, coords)[:3].T
    vox_coords = np.round(vox_coords).astype(int)
    return vox_coords


def _check_coord_inputs(coords):
    """
    Confirms `coords` are appropriate shape for coordinate transform

    Parameters
    ----------
    coords : array-like

    Returns
    -------
    coords : (4 x N) numpy.ndarray
    """
    coords = np.atleast_2d(coords).T
    if 3 not in coords.shape:
        raise ValueError('Input coordinates must be of shape (3 x N). '
                         'Provided coordinate shape: {}'.format(coords.shape))
    if coords.shape[0] != 3:
        coords = coords.T
    # add constant term to coords to make 4 x N
    coords = np.row_stack([coords, np.ones_like(coords[0])])
    return coords


def get_atlas(atlastype, cache=True):
    """
    Gets `atlastype` image and corresponding label file from package resources

    Parameters
    ----------
    atlastype : str
        Name of atlas to query
    cache : bool, optional
        Whether to pre-load atlas image data. Default: True

    Returns
    -------
    info : sklearn.utils.Bunch
        image : Niimg_like
            ROI image loaded with integer-based labels indicating parcels
        labels : pandas.core.data.DataFrame
            Dataframe with columns ['index', 'name'] matching region IDs in
            `image` to anatomical `name`
    """
    # we accept uppercase atlases via argparse but our filenames are lowercase
    atlastype = atlastype.lower()

    # get the path to atlas + label files shipped with package
    # resource_filename ensures that we're getting the correct path
    data_dir = resource_filename('atlasreader', 'data/atlases')
    atlas_path = op.join(data_dir, 'atlas_{0}.nii.gz'.format(atlastype))
    label_path = op.join(data_dir, 'labels_{0}.csv'.format(atlastype))

    if not all(op.exists(p) for p in [atlas_path, label_path]):
        raise ValueError('{} is not a valid atlas. Please check inputs and '
                         'try again.'.format(atlastype))

    atlas = Bunch(atlas=atlastype,
                  image=image.load_img(atlas_path).get_data(),
                  labels=pd.read_csv(label_path))
    if cache:
        atlas.image.get_fdata()

    return atlas


def get_label(atlastype, label_id):
    """
    Gets anatomical name of `label_id` in `atlastype`

    Parameters
    ----------
    atlastype : str
        Name of atlas to use
    label_id : int
        Numerical ID representing label

    Returns
    ------
    label : str
        Neuroanatomical region of `label_id` in `atlastype`
    """
    labels = check_atlases(atlastype).labels
    try:
        #print(atlastype)
        #print('test:', labels.query('index == {}'.format(label_id)).name.iloc[0])
        return labels.query('index == {}'.format(label_id)).name.iloc[0]
    except IndexError:
        return 'no_label'


def check_atlas_bounding_box(voxIDs, box_shape):
    """
    Returns the provided voxel ID if the voxel is inside the bounding box of
    the atlas image, otherwise the voxel ID will be replaced with the origin.

    Parameters
    ----------
    voxIDs : (N, 3) numpy.ndarray
        `coords` in cartesian space
    box_shape : (3,) list of int
        size of the atlas bounding box

    Returns
    ------
    ijk : (N, 3) numpy.ndarray
        `coords` in cartesian space that are inside the bounding box
    """

    # Detect voxels that are outside the atlas bounding box
    vox_outside_box = np.sum(
        (voxIDs < 0) + (voxIDs >= box_shape[:3]), axis=-1, dtype='bool')

    # Set those voxels to the origin (i.e. a voxel outside the brain)
    voxIDs[vox_outside_box] = np.zeros(3, dtype='int')

    return voxIDs
