import argparse
import os
import shutil
import logging
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from src.utils.common import read_yaml, create_directories, get_df
from src.utils.featurize import save_matrix


STAGE = "Two" 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    
    artifacts = config["artifacts"]
    prepared_data_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["PREPARED_DATA"])
    train_data_path = os.path.join(prepared_data_dir_path, artifacts["TRAIN_DATA"])
    test_data_path = os.path.join(prepared_data_dir_path, artifacts["TEST_DATA"])
    
    featurized_data_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["FEATURIZED_DATA"])
    create_directories([featurized_data_dir_path])

    featurized_train_data_path = os.path.join(prepared_data_dir_path, artifacts["FEATURIZED_OUT_TRAIN"])
    featurized_test_data_path = os.path.join(prepared_data_dir_path, artifacts["FEATURIZED_OUT_TEST"])

    max_features = params["featurize"]["max_features"]
    ngrams = params["featurize"]["ngrams"]

    # Read the data
    df_train = get_df(train_data_path)

    # Get the training words
    train_words = np.array(df_train.text.str.lower().values.astype("U"))

    # Generate the Feature Matrix
    bag_of_words = CountVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, ngrams)
    )

    bag_of_words.fit(train_words)
    train_words_binary_matrix = bag_of_words.transform(train_words)

    tfidf = TfidfVectorizer(smooth_idf=False) # smooth_idf: add 1 to denominator in idf
    tfidf.fit(train_words_binary_matrix)
    train_words_binary_matrix = tfidf.transform(train_words_binary_matrix)

    save_matrix(df_train, train_words_binary_matrix, featurized_train_data_path)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e
