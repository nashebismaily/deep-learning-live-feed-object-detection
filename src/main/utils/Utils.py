import os

# Utils is a utility class with helper functions which enable ease of use
class Utils:

    # return working directory of file
    @staticmethod
    def get_current_dir(file):
        return os.path.dirname(os.path.dirname(file))

    # return parent directory of file
    @staticmethod
    def get_parent_dir(path,index=0):
        parent_dir = path
        while index > 0:
            parent_dir = os.path.dirname(parent_dir)
            index -= 1
        return parent_dir

    # create absolute path of file
    @staticmethod
    def get_relative_path(path, index=0, file=None):
        parent_dir = path
        while index > 0:
            parent_dir = os.path.dirname(parent_dir)
            index -= 1
        if file:
            parent_dir = parent_dir + '/' + file
        return parent_dir

