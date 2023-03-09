import numpy as np
from mpl_toolkits.mplot3d import art3d

# adapted from https://stackoverflow.com/questions/18228966/how-can-matplotlib-2d-patches-be-transformed-to-3d-with-arbitrary-normals

def rotation_matrix(azim:int, elev:int, roll:int,):
    """
    Calculates the rotation matrix that changes v1 into v2.
    """
    # General 3d-rotation matrix for Tait–Bryan angles
    # https://en.wikipedia.org/wiki/Rotation_matrix#General_rotations
    # for vertical axis = z, order/role of angle changes, when vertical axis changes
    cosa, sina = np.cos(np.deg2rad(azim)), np.sin(np.deg2rad(azim))  # yaw, alpha  y
    cosb, sinb = np.cos(np.deg2rad(elev)), np.sin(np.deg2rad(elev))  # pitch, beta
    cosg, sing = np.cos(np.deg2rad(roll)), np.sin(np.deg2rad(roll))  # roll, gamma

    M = np.array([[cosa * cosb, cosa * sinb * sing - sina * cosg, cosa * sinb * cosg + sina * sing],
                [sina * cosb, sina * sinb * sing + cosa * cosg, sina * sinb * cosg - cosa * sing],
                [-sinb, cosb * sing, cosb * cosg]])
    return M        


def pathpatch_2d_to_3d(pathpatch, z:int, azim:int, elev:int, roll:int):
    """
    Transforms a 2D Patch to a 3D patch using the given normal vector.

    The patch is projected into they XY plane, rotated about the origin
    and finally translated by z.
    """
    path = pathpatch.get_path()  # Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path)  # Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D  # Change the class
    pathpatch._code3d = path.codes  # Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor  # Get the face color    

    verts = path.vertices  # Get the vertices in 2D

    M = rotation_matrix(azim, elev, roll)  # Get the rotation matrix
    pathpatch._segment3d = np.array([np.dot(M, (x, y, z)) for x, y in verts])


def pathpatch_translate(pathpatch, delta):
    """
    Translates the 3D pathpatch by the amount delta.
    """
    pathpatch._segment3d += delta