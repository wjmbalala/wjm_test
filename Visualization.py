import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# set saved model path
setting = 'informer_ETTh1_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0'

preds = np.load('./results/'+setting+'/pred.npy')
trues = np.load('./results/'+setting+'/true.npy')

# draw OT prediction
plt.figure()
plt.plot(trues[0, :, -1], label='GroundTruth')
plt.plot(preds[0, :, -1], label='Prediction')
plt.legend()
plt.show()
