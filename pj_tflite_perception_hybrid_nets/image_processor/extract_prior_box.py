import numpy as np

prior = np.load('anchors_384x640.npy')[0]

for prior_box in prior[:,]:
    print(str(prior_box[0]) + ", " + str(prior_box[1]) + ", " + str(prior_box[2]) + ", " + str(prior_box[3]) + ", ")

