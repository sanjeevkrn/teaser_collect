## News Article teaser tweet collector.

Version: 0.2

Downloads live tweets from a given list of Twitter accounts and downloads the news articles that the tweets link to. And, scripts to extract teasers from downloaded tweets.

### Dependencies

* newspaper, for parsing text content
* tweepy, for reading the Twitter stream


### Installation

    $ git clone ...
    $ python setup.py develop
    $ pip install -r requirements.txt

Then you have to add authorisation credentials for accessing the Twitter API. Please add the necessary details for your Twitter account in the file src/main/python/twitter_data_collection/etc/twitter_auth.conf, or specify an alternative config file in the same format at the command line.

### Usage

	$ twitter_read_stream -h
	usage: twitter_read_stream [-h] [-v] [-o OUTPUT_FILE] [-e HTML_OUTPUT_FILE] [-c CONF_FILE]
	                           [-l LOGFILE] [-p]
	
	optional arguments:
	  -h, --help            show this help message and exit
	  -v, --version         show program's version number and exit
	  -o OUTPUT_FILE, --output_file OUTPUT_FILE
	                        file to write data to
	  -e HTML_OUTPUT_FILE, --html_file HTML_OUTPUT_FILE
      						file to write html to
      -c CONF_FILE, --conf_file CONF_FILE
	                        Config file location
	  -l LOGFILE, --logfile LOGFILE
	                        Log file location
	  -p, --append          append to given output files


### Output

A file in which each line contains a pair of tweet and article content. Another file in which each line has tweet id, screen name, article url and html of the article. The files format is one-json-object-per-line.

### Running in a container

After building the container, you run it by mounting your authorisation credentials in a file into the appropriate location. Data is written into the folder `/data`, which you might like to mount from the host.

```
docker run -d --rm -v some_volume:/data -v my_auth.conf:/auth/twitter_auth.conf <image_name>
```

### convert json data to pandas dataframe
```
python convert_data_to_pandas.py -i some_volume/twitter_data_${TIMESTAMP}.json -o some_volume/data_tweet.pkl
```
### compile teasers from pandas dataframe
As several preprocessing steps are applied to dataframe (data_tweet.pkl), if dataframe is large, completion of the below command can be very time consuming.
We recommend to execute each region of code inside method main individually.
```
python study_of_data.py -i some_volume/data_tweet.pkl -o some_volume/processed_dataset
```
### our NAACL train, eval and test dataset are at:
https://s3-eu-west-1.amazonaws.com/teasers.naacl19/teaser_base_naacl19.tar.gz

To train baseline system, please check [this repository](https://github.com/sanjeevkrn/teaser_generate).

For more details, check out our paper:
- SK Karn, M Buckley, U Waltinger, H Sch{\"u}tze. [News Article Teaser Tweets and How to Generate Them](https://www.aclweb.org/anthology/N19-1398.pdf).

## How do I cite this work?
```
@inproceedings{karn-etal-2019-news,
    title = "News Article Teaser Tweets and How to Generate Them",
    author = {Karn, Sanjeev Kumar  and
      Buckley, Mark  and
      Waltinger, Ulli  and
      Sch{\"u}tze, Hinrich},
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N19-1398",
    doi = "10.18653/v1/N19-1398",
    pages = "3967--3977",
}
```
