import joblib
import logging
import numpy as np
import scipy.sparse as sparse

def save_matrix(df, matrix, out_path):
    id_matrix = sparse.csr_matrix(df.id.astype(np.int64)).T
    label_matrix = sparse.csr_matrix(df.label.astype(np.int64)).T

    result = sparse.hstack([id_matrix, label_matrix, matrix], format="csr")

    msg = f"The ouput matrix {out_path} of size {result.shape} and data type: {result.dtype}"
    logging.info(msg)
    joblib.dump(result, out_path)
