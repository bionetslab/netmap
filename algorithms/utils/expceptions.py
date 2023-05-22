class ColumnValidationError(Exception):
    """
    Raise this exception if the Column names in the node and edge files are not
    named correctly.
    """

    pass


class NotInitializerError(Exception):
    """
    Raise this exception, if a mandatory data element has not been intialized.


    """

    pass


class InconsistenClusterExpection(Exception):
    """

    This exception is raised if the clusters are not consistent between several
    runs.
    """

    pass


class ConfigValidationError(Exception):
    """
    Raise this Exception if the configuration file is incomplete and cannot be 
    automatically filled.
    """

    pass