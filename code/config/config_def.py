TRAIN = 'train'
VALIDATION = 'val'
TEST = 'test'

SUBSETS = [TRAIN, VALIDATION, TEST]

def verifySubset(subset):
  if not subset in SUBSETS:
    raise Exception('invalid argument %s that not in %s' % (subset, SUBSETS))

# ----
OBJECTIVE_TYPE_CATEGORICAL = 'categorical'
OBJECTIVE_TYPE_REGRESSION = 'regression'
OBJECTIVE_TYPE_BINARY = 'binary'

OBJECTIVE_TYPES = [
  OBJECTIVE_TYPE_CATEGORICAL,
  OBJECTIVE_TYPE_REGRESSION,
  OBJECTIVE_TYPE_BINARY
]

def verifyObjectiveType(t):
  if not t in OBJECTIVE_TYPES:
    raise Exception('invalid argument %s that not in %s' % (t, OBJECTIVE_TYPES))

