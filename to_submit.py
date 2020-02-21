import numpy as np
import pandas as pd

if __name__ == '__main__':
    d = np.genfromtxt('preds.csv', delimiter=',')
    d = np.int64(d)
    xx = {'Id': np.arange(len(d)), 'Digit': d}
    df = pd.DataFrame(xx)
    df.to_csv('submission2.csv', index=False)