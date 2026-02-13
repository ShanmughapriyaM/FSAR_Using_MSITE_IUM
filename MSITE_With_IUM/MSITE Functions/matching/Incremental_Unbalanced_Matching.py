import os
import numpy as np
from collections import defaultdict
from temporal_encoding.segmented_multiscale import segmented_multiscale_encoding

feat = np.load("ExtractedFeatures/FeatureSet1/Subject.9_Walk_WalkS9.npy")

query_vector = segmented_multiscale_encoding(
    feat,
    scales=(1, 2, 4)
)

print(query_vector)