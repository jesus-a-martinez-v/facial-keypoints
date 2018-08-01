def build_path(segment, running_on_floydhub=False):
    """
	Builds the full path to `segment`, depending on where we are running our code.

	Args
		:segment File or directory we want to build the full path to.
	"""
    
    if running_on_floydhub:
        return '/floyd/data/{}'.format(segment)
    else:
        return 'data/{}'.format(segment)