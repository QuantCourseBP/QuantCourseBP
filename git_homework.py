from importlib import metadata
import json
import numpy as np
import pandas as pd
import sys

from src import package_location, HOMEWORK_DIR, GIT_DATA

GIT_HMWK_SOURCE = package_location.joinpath(HOMEWORK_DIR)


def get_pip_info():
    packages = {item.metadata['Name']: item.version for item in metadata.distributions()}
    packages['PYTHON'] = sys.version
    # convert dictionary to JSON string
    json_string = json.dumps(packages, indent=4)
    # write env info into file
    GIT_HMWK_SOURCE.joinpath('my_env.json').write_text(json_string)


def get_git_info():
    my_config = json.loads(GIT_HMWK_SOURCE.joinpath('my_config.json').read_text())
    assert my_config['course_id'], "Feed your course id into the 'my_config.json' first."
    GIT_DATA['user'] = my_config['course_id']
    GIT_HMWK_SOURCE.joinpath('my_git.json').write_text(json.dumps(GIT_DATA, indent=4))


if __name__ == '__main__':
    get_pip_info()
    get_git_info()

    a = pd.read_csv(GIT_HMWK_SOURCE.joinpath('data.csv')).to_numpy()
    b = np.linalg.inv(a)
    np.savetxt(GIT_HMWK_SOURCE.joinpath('inv.csv'), b, fmt='%.4f', delimiter=',')
    print('Done.')
