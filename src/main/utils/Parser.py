from src.main.utils.PathBuilder import PathBuilder

# Utils is a utility class with helper functions which enable ease of use
class Parser:

    @staticmethod
    def get_config(parser, config, section, index=4):
        parser.read(PathBuilder.get_relative_path(__file__, index, 'config/' + config))
        return dict(parser.items(section))

