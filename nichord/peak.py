'''
This code has been adapted from atlasreader: https://github.com/miykael/atlasreader
atlasreader has not been uploaded to conda-forge, and so parts of its code were
    copied here, so that NiChord can be uploaded to conda-forge.
'''

import numpy as np


def _read_atlas_peak(atlastype, coordinate):
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

    try:
        data = atlastype.image.get_fdata()
        # get voxel index
    except:
        data = atlastype.image

    voxID = _coord_xyz_to_ijk(atlastype.image.affine, coordinate).squeeze()
    voxID = _check_atlas_bounding_box(voxID, data.shape)
    labelID = int(data[voxID[0], voxID[1], voxID[2]])
    label = _get_label(atlastype, labelID)
    return label


def _coord_xyz_to_ijk(affine, coords):
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



def _get_label(atlastype, label_id):
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
    labels = atlastype.labels
    try:
        #print(atlastype)
        #print('test:', labels.query('index == {}'.format(label_id)).name.iloc[0])
        return labels.query('index == {}'.format(label_id)).name.iloc[0]
    except IndexError:
        return 'no_label'


def _check_atlas_bounding_box(voxIDs, box_shape):
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
