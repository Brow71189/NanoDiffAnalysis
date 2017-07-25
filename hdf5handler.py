import h5py

def openhdf5file(hdf5filename):
    hdf5filelink=h5py.File(hdf5filename, 'r')
    return hdf5filelink

def gethdf5slice(slicenumber,hdf5filelink):
    hdf5slice = hdf5filelink['data/science_data/data'][slicenumber,:]
    return hdf5slice
