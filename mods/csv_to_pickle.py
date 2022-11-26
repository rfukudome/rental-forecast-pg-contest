import os.path as path
import pandas as pd
from logging import getLogger


def csv_to_picle(dir,file_name):
    logger = getLogger(__name__)
    logger.info('csvファイルをpickleファイルに変換')
    PATH_CSV_FILE = path.join(dir, file_name+'.csv')
    PATH_PICKLE_FILE = path.join('input',file_name+'.pickle')
    df = pd.read_csv(PATH_CSV_FILE, encoding="utf-8")
    pickle_file = df.to_pickle(PATH_PICKLE_FILE)
    logger.info('pickleファイルに変換完了しました')

    return pickle_file