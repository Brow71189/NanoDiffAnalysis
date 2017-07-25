from . import hdf5handler as h5f
import numpy as np

def getroirange(hdf5filelink,roi):
    """
    input:
    hdf5filelink: link to a hdf5 file (<HDF5 file "filename.h5" (mode r+)>)
    roi: dict containing ROI data ({'center': (y,x), 'size': (h,w)})
    output:
    roirange: dict containing pixel range of rois
    """
    y,x = h5f.gethdf5slice(0,hdf5filelink).shape
    roirely, roirelx = roi['center']
    roirelh, roirelw = roi['size']
    # top left and bottom right of the roi
    roiymin = int((roirely - roirelh * 0.5) * y)
    roixmin = int((roirelx - roirelw * 0.5) * x)
    roiymax = int((roirely + roirelh * 0.5) * y)
    roixmax = int((roirelx + roirelw * 0.5) * x)
    roirange = {"roiymin": roiymin, "roiymax": roiymax, "roixmin": roixmin,
            "roixmax": roixmax}
    return roirange


def vdf(hdf5filelink,roirange):
    """
    input:
    hdf5filelink: link to a hdf5 file (<HDF5 file "filename.h5" (mode r+)>)
    roirange: dict containing pixel range of rois
    output:
    numpyarray of vdf
    """
    ymin = roirange['roiymin']
    ymax = roirange['roiymax']
    xmin = roirange['roixmin']
    xmax = roirange['roixmax']
    a = hdf5filelink['data/science_data/data'][:,ymin:ymax,xmin:xmax]
    a = np.average(a, axis=1)
    a = np.average(a, axis=1)
    #vdfx =int(1+math.sqrt(1+c.shape[0]))
    vdfy = int(np.sqrt(a.shape[0]))
    vdfx = int(a.shape[0]/vdfy)
    a = np.reshape(a[:vdfx*vdfy], (vdfy,vdfx))
    return a
