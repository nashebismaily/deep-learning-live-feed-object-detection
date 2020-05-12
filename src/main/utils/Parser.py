from src.main.utils.PathBuilder import PathBuilder

# Utils is a utility class with helper functions which enable ease of use
class Parser:

    @staticmethod
    def get_config(parser, config, section, index=4):
        parser.read(PathBuilder.get_relative_path(__file__, index, 'config/' + config))
        return dict(parser.items(section))

    @staticmethod
    def string_to_bool(string):
        if string == 'True':
            return True
        elif string == 'False':
            return False
        else:
            raise ValueError