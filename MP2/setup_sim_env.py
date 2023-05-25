"""Clone all FARMS repos"""

import os
import sys
from subprocess import check_call
try:
    from git import Repo
except ImportError:
    check_call([sys.executable, '-m', 'pip', 'install', 'GitPython'])
    from git import Repo



def main():
    """Main"""
    # pip_install = [sys.executable, '-m', 'pip', 'install']
    pip_install = ['pip', 'install']

    # Install MuJoCo
    check_call(pip_install + ['mujoco'])

    check_call(pip_install + ['dm_control'])

    # FARMS
    for package in ['farms_core', 'farms_mujoco', 'farms_sim']:
        print(f'Providing option to reinstall {package} if already installed')
        check_call(['pip', 'uninstall', package])
        print(f'Installing {package}')
        check_call(
            pip_install
            + [
                '--no-cache-dir',
                f'https://gitlab.com/farmsim/{package}/'
                f'-/archive/cmc_2022/{package}-cmc_2022.zip',
                # '-vvv',
            ]
        )
        print(f'Completed installation of {package}\n')


if __name__ == '__main__':
    main()
