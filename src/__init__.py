import os
from pathlib import Path


package_location = Path(__file__).parent.parent.resolve()
package_root = os.path.dirname(__file__)


GIT_DATA = {}


def _set_git_data():
    global GIT_DATA
    import git
    this_repo = git.Repo(str(package_location))
    GIT_DATA['repository'] = this_repo.remotes.origin.url
    try:
        GIT_DATA['branch'] = this_repo.active_branch.name
    except:
        GIT_DATA['branch'] = 'detached_' + this_repo.head.object.hexsha
    
    
    quantc_hash = this_repo.commit().hexsha
    quantc_dirty = '*' if this_repo.is_dirty(untracked_files=True) else ''
    
    GIT_DATA['sha'] = f'QuantCourseBP {quantc_hash}{quantc_dirty}'
    print(GIT_DATA['sha'])


_set_git_data()
