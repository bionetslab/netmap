

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