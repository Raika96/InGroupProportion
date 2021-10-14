from scipy.spatial.distance import cdist
import numpy as np

# corr determine the measure for finding the neighbour. The options are Euclidean distance and Pearson correlation.
def calculateIGP(data, labels, corr=True):

    dataNeighbour = np.zeros(labels.shape[0])
    if corr:
        correlation = np.corrcoef(data, rowvar = False)
        correlation[np.arange(dist.shape[0]), np.arange(dist.shape[0])] = np.nan
        dataNeighbour =  np.nanargmax(correlation, axis=1)
    else:
        dist = cdist(data, data, 'euclidean')
        dist[np.arange(dist.shape[0]), np.arange(dist.shape[0])] = np.nan
        dataNeighbour =  np.nanargmin(dist, axis=1)

    neighbourLabel = [labels[int(x)] for x in dataNeighbour]
    difLabel = labels - neighbourLabel
    numZeros = difLabel[difLabel==0].shape[0]
    igp = numZeros/difLabel.shape[0]

    return igp
