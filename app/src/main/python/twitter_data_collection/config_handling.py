
import configparser

from pkg_resources import resource_stream


def load_config(config_filename=None):
    """
    Reads the config file and adds any settings in it to the default settings

    :param config_filename:
    :return: A ConfigParser object full of settings
    """
    config = configparser.SafeConfigParser()

    # first, read the defaults
    if config_filename:
        config.read(config_filename)
    else:
        config.readfp(resource_stream('twitter_data_collection.etc', config_filename))

    return config
