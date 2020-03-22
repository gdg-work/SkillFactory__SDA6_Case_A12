#!/usr/bin/env python3

import os
import fnmatch

# builders
Py_2_NB = Builder(
    action = Action(["jupytext --to ipynb --from py:percent  $SOURCE"]),
    # action = Action(["pipenv run jupytext --to ipynb --from py:percent  $SOURCE"]),
    src_suffix = ".py",
    suffix = ".ipynb")

# environment
env = Environment(ENV = {'PATH' : os.environ['PATH'] },
                  platform = 'posix',
                  SrcDir   = ".",
                  Out_Dir  = ".",
                  BUILDERS = { "Python":Py_2_NB },
                  )

env.Depends('Case_A12_1.ipynb', 'Case_A12_1.py')
env.Depends('Case_A12_2.ipynb', 'Case_A12_2.py')

env.Default('Case_A12_2.ipynb')
env.Python(source='Case_A12_2.py')
