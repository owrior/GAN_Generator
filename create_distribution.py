import pandas as pd
import numpy as np

class Dgen:
    def __init__(self, nstores, nprods, nrecords):
        self._nstores = nstores
        self._storenum = 0
        self._nprods = nprods
        self._nrecords = nrecords
        self._df = pd.DataFrame(np.nan, index=range(0, nrecords * nstores), 
            columns=['store', 'product'])

    def generate_store(self):
        probs = np.random.normal(0, 1, self._nrecords)
        self._df.loc[int(self._storenum * self._nrecords):int(self._storenum * self._nrecords + self._nrecords - 1),
            ['store', 'product']] = "".join(['store_', str(self._storenum)]), \
                pd.cut(probs, self._nprods, labels = ['product_' + str(x) for x in range(1, self._nprods + 1)])
        self._storenum += 1


    def generate_data(self):
        for i in range(0, self._nstores):
            self.generate_store()