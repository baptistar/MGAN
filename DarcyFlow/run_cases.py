import numpy as np
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

latent_dim = [25]
monotone_grid = [1e-2]
stdy_grid = [0.0002,0.0005]

for ldim in latent_dim:
    for monotone_param in monotone_grid:
        for stdy in stdy_grid:
            # define str for test case
            strval='python wgan_darcy.py --m_lambda ' + str(monotone_param) + ' --latent_dim ' + str(ldim) + ' --std_y ' + str(stdy)
            # run example
            os.system(strval)

