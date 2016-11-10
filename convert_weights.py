""" Convert weights from matconvnet model to an hdf5 dataset loadable into Lasagne models.

Call with ./matconv2hdf5.py [basename]
"""

import sys, os
from urllib.request import urlretrieve
import scipy.io
import collections
import h5py

def load_mat(basename):
    weights = scipy.io.loadmat(basename + ".mat")
    w_dict = collections.OrderedDict()

    for param in weights["params"].flatten():
        for p in param:
            if p.dtype == "float32":
                #print(param[0][0], p.shape, p.dtype)
                w_dict[param[0][0]] = p

    return w_dict

def remap_weights(d):
    d_ = collections.OrderedDict()
    for k in d.keys():
        parts = k.split("_")
        
        if len(parts) == 3:
            layer, location, wtype = parts
            layer = layer + "_" + location + "/"
        else:
            layer, wtype = parts
            layer = layer + "/"
            
        rename = {"filter" : "W", "bias": "b"}
            
        if "bn" in layer:
            rename = {"mult":"gamma", "bias":"beta", "moments":""}
            new_key = layer+rename[wtype]
            if wtype == "moments":
                d_[layer+"mean"] = d[k][:,0]
                
                d_[layer+"inv_std"] = 1/d[k][:,1] + 1e-4 
            else:
                d_[new_key] = d[k][:,0]
        else:
            new_key = layer+rename[wtype]
            if wtype == "filter":
                if "fc" in layer:
                    d_[new_key] = d[k][0,0,...]
                else:
                    d_[new_key] = d[k].transpose((3,2,1,0))
            elif wtype == "bias":
                d_[new_key] = d[k][:,0]
            else:
                d_[new_key] = d[k]

    return d_

if __name__ == '__main__':
    if not len(sys.argv) == 2:
        print("Usage:", sys.argv[0], "[basename]")
        print("")
        print("Basename is the name taken from http://www.vlfeat.org/matconvnet/pretrained")
        sys.exit(1)

    basename = sys.argv[1]
    print("Basename", basename)

    if not os.path.exists(basename + ".mat"):
        print("Retrieving model: ", basename)
        urlretrieve ("http://www.vlfeat.org/matconvnet/models/"+basename+".mat", basename + ".mat")
    else:
        print("Found model in current path.")
    print("Done.")


    # Load weights from the .mat dataset and display names + shapes
    w_dict = load_mat(basename)
    print(w_dict.keys())

    # remap and format the weights to be able to load them into Lasagne
    w_ = remap_weights(w_dict)
    
    # write HDF5 dataset
    if os.path.exists(basename + ".hdf5"):
        if input("File exists. Overwrite? (y/n)") != "y":
            print("Exit")
            sys.exit(1)
        os.unlink(basename + ".hdf5")
    else:
        with h5py.File(basename + ".hdf5") as ds:
            for k in w_.keys():
                ds.create_dataset(k, shape=w_[k].shape, data=w_[k], dtype="float32")

        print("Done. File saved to ", basename+".hdf5") 

