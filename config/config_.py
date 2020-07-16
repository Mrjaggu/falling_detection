from configparser import ConfigParser
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

configur = ConfigParser()
configur.read('../config/config.ini')
dataset_path = configur.get('{}'.format('dataset'), 'Dataset')
folder_path = configur.get('{}'.format('dataset'), 'folder')