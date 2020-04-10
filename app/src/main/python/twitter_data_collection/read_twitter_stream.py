import argparse
import codecs
import json
import logging
import logging.handlers
import logging.config
from pkg_resources import resource_stream
import os
from newspaper import Article

import time
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

# from  config_handling import load_config
# from __init__ import __version__ as version
from twitter_data_collection.config_handling import load_config
from twitter_data_collection import __version__ as version

# See https://marcobonzanini.com/2015/03/02/mining-twitter-data-with-python-part-1/


def setup_argparser():
    my_parser = argparse.ArgumentParser(prog='twitter_read_stream')

    my_parser.add_argument('-v', '--version', action='version', version='Twitter data collector v%s' % version)
    my_parser.add_argument("-o",
                           "--output_file",
                           type=str,
                           help="file to write data to",
                           default=None)
    my_parser.add_argument("-e",
                           "--html_file",
                           type=str,
                           help="file to write html to",
                           default=None)
    my_parser.add_argument("-c",
                           "--conf_file",
                           type=str,
                           help="Config file location",
                           default=None)
    my_parser.add_argument("-l",
                           "--logfile",
                           type=str,
                           help="Log file location",
                           default='twitter.log')
    my_parser.add_argument("-p",
                           "--append",
                           help="append to given output files",
                           action="store_true")
    return my_parser


class WriteToFileListener(StreamListener):
    """
    This is a basic listener that just prints received tweets to stdout.
    """

    def __init__(self, outfile, htmlfile, ids, append):
        """

        :param outfile:
        :param ids: the list of ids we're interested in
        :return:
        """
        self.logger = logging.getLogger(__name__)
        if append:
            self.logger.info("Append to files.")
            modestring = 'a'
        else:
            modestring = 'w'

        self.output_file_handle = codecs.open(outfile, modestring, encoding='utf-8')
        self.html_file_handle = codecs.open(htmlfile, modestring, encoding='utf-8')
        self.ids = ids
        self.tweet_counter = 0
        super(WriteToFileListener, self).__init__()

    def on_status(self, status):
        """
        Don't store retweets. Only keep tweets from our list of accounts.

        :param status:
        :return:
        """
        try:
            if not status.retweeted and status.user.id_str in self.ids:
                self.tweet_counter += 1
                self.logger.info("Tweet counter: %s" % self.tweet_counter)
                self.logger.info('%s %s: %s' % (status.id, status.user.screen_name, status.text))

                orig_tweet = status._json
                # url_struct = status.entities['urls'][0]

                if 'retweeted_status' in orig_tweet:
                    self.logger.info("retweeted_status......................")
                    tweet_fnl = orig_tweet['retweeted_status']
                else:
                    tweet_fnl = orig_tweet
                if 'extended_tweet' in tweet_fnl:
                    self.logger.info("extended_tweet......................")
                    urls = tweet_fnl['extended_tweet']['entities']['urls']
                else:
                    urls = tweet_fnl['entities']['urls']
                tweet_id = tweet_fnl['id']
                tweet_screen_name = tweet_fnl['user']['screen_name']
                if len(urls) == 0:
                    self.logger.info("Empty url_struct for id %s and user %s.\n" % (tweet_id, tweet_screen_name))
                    return True

                url_struct = urls[0]
                url = url_struct['url']
                article_content, html_b64 = self.parse_article_from_url(url, tweet_id)

                output = {
                    'tweet': tweet_fnl,
                    'text': article_content
                }
                html_out = {
                    'tweet_id': tweet_id,
                    'tweet_screen_name': tweet_screen_name,
                    'url': url,
                    'html_article': html_b64
                }
                try:
                    self.output_file_handle.write(json.dumps(output))
                    self.html_file_handle.write(json.dumps(html_out))
                except Exception as inst:
                    self.logger.info("Error %s while dumping json.\n" % inst)
                    return True
                self.output_file_handle.write('\n')
                self.html_file_handle.write('\n')
                self.output_file_handle.flush()
                self.html_file_handle.flush()

                self.logger.info("Finished retrieval process for url: %s\n" % url)
                return True
        except Exception as inst:
            self.logger.info("Error %s while processing the tweet. Skipping.\n" % inst)
            return True

    def on_error(self, status_code):
        self.logger.info('Error code: %s' % status_code)
        if status_code == 420:
            # we're being rate limited, so let's take a 30 min break
            time.sleep(30*60)

        # Keep the connection open even though there was an error
        return True

    def parse_article_from_url(self, url, tweet_id):
        """
        downloads and parses an article to json
        :param url:
        :param tweet_id:
        :return:
        """

        try:
            a = Article(url)

            a.download()
            article_html = a.html
            a.parse()
            text = a.text
            title = a.title
            meta_data = a.meta_data
            is_media_news = a.is_media_news()
            is_parsed = a.is_parsed
            is_downloaded = a.download_state
            authors = a.authors
            canonical_link = a.canonical_link
            is_valid = True
            is_reloaded = False

        except Exception as inst:
            self.logger.info("Error %s while loading and parsing article for tweet id %s with url %s" % (inst, tweet_id, url))
            article_html = ''
            text = ''
            title = ''
            meta_data = ''
            is_media_news = ''
            is_parsed = ''
            is_downloaded = ''
            authors = ''
            canonical_link = ''
            is_valid = False
            is_reloaded = False

        result = {#'html_b64': article_html,
                  'text_b64': text,
                  'url': url,
                  'id': tweet_id,
                  'title': title,
                  'meta_data': meta_data,
                  'is_media_news': is_media_news,
                  'publish_date': '',
                  'is_parsed': is_parsed,
                  'is_downloaded': is_downloaded,
                  'authors': authors,
                  'canonical_link': canonical_link,
                  'is_valid': is_valid,
                  'is_reloaded': is_reloaded}
        # result = {
        #         'html_b64': article_html,
        #         'text_b64': text,
        #         'url': url,
        #         'id': tweet_id,
        #         'title': title
        #     }

        return result, article_html


def main():

    parser = setup_argparser()
    args = parser.parse_args()

    conf = load_config(args.conf_file)
    consumer_key = conf.get('auth', 'consumer_key')
    consumer_secret = conf.get('auth', 'consumer_secret')
    access_token = conf.get('auth', 'access_token')
    access_secret = conf.get('auth', 'access_secret')

    # set up logging from the conf file
    # log_conf_path = resource_stream('etc', 'logging.conf')
    # log_conf_path = os.path.join('etc', 'logging.conf')
    log_conf_path = os.path.join(os.getcwd(), 'src/main/python/twitter_data_collection', 'etc', 'logging.conf')
    print(log_conf_path,    os.path.exists(log_conf_path))
    logging.config.fileConfig(log_conf_path, defaults={'logfilename': args.logfile})
    logger = logging.getLogger(__name__)

    # "follow" takes a list of ids, which is not the username.
    follow_list = [
        ### US
        '759251',  # CNN
        '807095',  # nytimes
        '35773039',  # theatlantic
        '14677919',  # newyorker
        '14511951',  # HuffingtonPost
        '1367531',  #FoxNews
        '28785486',  # ABC
        '14173315',  #NBCNews
        '2467791',  #washingtonpost
        '14293310',  #TIME
        '2884771',  #Newsweek
        '15754281',  #USATODAY
        '16273831',  #VOANews
        '3108351',  #WSJ
        '14192680',  #NOLAnews
        '15012486',  #CBSNews
        '12811952',  #Suntimes
        '14304462',  #TB_Times
        '8940342',  #HoustonChron
        '16664681',  #latimes
        '14221917',  #phillydotcom
        '14179819',  #njdotcom
        '15679641',  #dallasnews
        '4170491',  #ajc
        '6577642',  #usnews
        '1652541',  #reuters
        '9763482',  #nydailynews
        '17469289',  #nypost
        '12811952',  #suntimes
        '7313362',  #chicagotribune
        '8861182',  #newsday
        '17820493',  #ocregister
        '11877492',  #starledger
        '14267944',  #clevelanddotcom
        '14495726',  #phillyinquirer
        '17348525',  #startribune
        ### UK/Ireland
        '87818409',  # guardian
        '15084853',  # IrishTimes
        '34655603',  # thesun
        '15438913',  # mailonline
        '111556423',  # dailymailuk
        '380285402',  # dailymail
        '5988062',  # theeconomist
        '17680050',  # thescotsman
        '16973333',  # independent
        '17895820',  # daily_express
        # '18949452',  # ft
        # '4898091',  # financialtimes
        ### International
        '4970411',  # ajenglish

    ]

    # This handles Twitter authentification and the connection to Twitter Streaming API
    listener = WriteToFileListener(args.output_file, args.html_file, follow_list, args.append)
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    stream = Stream(auth, listener)

    logger.info("Starting to read Twitter stream...")
    stream.filter(follow=follow_list)


if __name__ == '__main__':

    while True:
        try:
            main()
        except Exception as err:
            print(err)
            print("there was a crash. restarting the stream reader...\n")

        time.sleep(60)
