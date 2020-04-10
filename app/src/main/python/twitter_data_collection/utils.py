# -*- coding: utf-8 -*-
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
import nltk
import pandas as pd
from html.parser import HTMLParser
from nltk.tokenize import sent_tokenize
from nltk.tokenize.casual import _replace_html_entities
from itertools import islice
from collections import Counter, OrderedDict
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import doc2vec

stemmer = SnowballStemmer("english")
lmtzr = WordNetLemmatizer()

hparser = HTMLParser()

stopwords_en = stopwords.words('english')


def remove_urls(text):
    # http, then anything which is not whitespace, up to a word boundary
    return re.sub(r'http[^\s]+', '', text)


def get_ngrams(text, n=2, remove_stopwords=False, stem=False):
    tokens = text.split()
    # tokens = word_tokenize(text)
    if remove_stopwords:
        tokens = [tok for tok in tokens if tok not in stopwords_en and len(tok) > 1]
    if stem:
        tokens = list(map(stemmer.stem, tokens))
        # tokens = map(lambda x: lmtzr.lemmatize(x, 'v'), map(lmtzr.lemmatize, tokens))
    if n == 1:
        return list(map(tuple, list(map(lambda x: [x], tokens))))
    else:
        return ngrams(tokens, n)


def percentage_match_tokens(s_seq, l_seq, stem=False):
    s_tokens = get_ngrams(s_seq, 1, True, stem)
    l_tokens = get_ngrams(l_seq, 1, True, stem)
    s_tokens_set = set(s_tokens)
    l_tokens_set = set(l_tokens)
    if len(l_tokens_set) == 0:
        return 0.0
    inters_size = len(s_tokens_set.intersection(l_tokens_set))
    if inters_size == 0:
        return 0.0
    return float(inters_size) / len(s_tokens_set)


def colname_to_value(cvg_tpl_lst):
    tr_cvg_dict = {}
    for tpls in cvg_tpl_lst:
        tpls_lst = list(tpls)
        colname = tpls_lst[0]
        cvg_val = tpls_lst[1][-1]
        val = colname.replace('tr_per_twt_', '')
        cvg_tr = float(val)/(10**len(val))
        tr_cvg_dict.update({cvg_tr: cvg_val})
    return sorted(tr_cvg_dict.items(), key=lambda itm:itm[0], reverse=True)


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def percentage_match_window_tokens(s_seq, l_seq, w_len, stem=False, min_match=0):
    s_tokens = get_ngrams(s_seq, 1, True, stem)  # s_seq = x['tokenized_tweet'].lower()
    s_tokens_set = set(s_tokens)
    sentences = sent_tokenize(l_seq)  # l_seq = x['tokenized_article'].lower()
    sentences_ngrams = [get_ngrams(sent, 1, True, stem) for sent in sentences]
    lst_ovlp = []
    win_len = min(w_len, len(sentences_ngrams))
    for i, wind_sent in enumerate(window(sentences_ngrams, win_len)):
        # l_tokens = get_ngrams(' '.join(wind_sent), 1, True, stem)
        l_tokens = [t for s in wind_sent for t in s]
        inters_size = len(s_tokens_set.intersection(set(l_tokens)))
        if inters_size <= min_match:
            lst_ovlp.append((i, 0.0))
            continue
        ovlp = float(inters_size) / len(s_tokens_set)
        lst_ovlp.append((i, ovlp))
    return tuple(sorted(lst_ovlp, key=lambda x: x[1], reverse=True))


def get_jaccard_similarity(headline, tweet, n=1):
    """
    tokenise to words, compute jaccard distance of word sets
    :param headline:
    :param tweet:
    :param n: n-gram length
    :return:
    """
    h_tokens = ngrams(word_tokenize(headline.lower()), n)
    t_tokens = ngrams(word_tokenize(tweet.lower()), n)
    return get_jaccard_similarity_tokens(h_tokens, t_tokens)


def get_jaccard_similarity_tokens(h_tokens, t_tokens):
    """
    Calculate the jaccard distance between the given sets of tokens
    :param h_tokens:
    :param t_tokens:
    :return:
    """
    h_tokens_set = set(h_tokens)
    t_tokens_set = set(t_tokens)
    inters_size = len(h_tokens_set.intersection(t_tokens_set))
    # print(inters_size)
    if inters_size == 0:
        return 0.0
    union_size = len(h_tokens_set.union(t_tokens_set))
    # print(union_size)
    if union_size == 0:
        return 0.0
    return float(inters_size) / union_size


def process_articles(row):
    lead = row['text.meta_data.description'] if 'text.meta_data.description' in row and not pd.isnull(
        row['text.meta_data.description']) else ''
    try:
        lead = hparser.unescape(lead)
    except Exception as e:
        print(lead)
        lead = ''
    title = row['text.title'] if not pd.isnull(row['text.title']) else ''
    article_raw = row['text.text_b64'] if not pd.isnull(row['text.text_b64']) else ''
    para_ = [para for para in article_raw.strip().split('\n\n') if len(para.strip()) > 0]
    if len(para_) > 0 and 'Story highlights' in para_[0]:
        para_ = para_[[i for i, p in enumerate(para_[:5]) if '(CNN)' in p][0]:] \
            if len([i for i, p in enumerate(para_[:5]) if '(CNN)' in p]) > 0 else para_
    para_clean = [para for para in para_ if
                  'click to share on whatsapp (opens in new window)' not in para.lower() and
                  'click to share on facebook (opens in new window)' not in para.lower() and
                  'click to share on twitter (opens in new window)' not in para.lower() and
                  'skip in skip x embed x share close' not in para.lower() and
                  '(photo' not in para.lower() and
                  'Photo:' not in para and
                  'PHOTOS:' not in para and
                  'Photograph:' not in para and
                  'file -' not in para.lower() and
                  '(AP Photo' not in para and
                  'AP Photo)' not in para and
                  '(Facebook)' not in para and
                  'Getty' not in para and
                  'GETTY' not in para and
                  'Play slideshow' not in para and
                  'MOST READ' not in para and  # TheSun
                  '(all times EDT)' not in para and
                  'Follow him on Twitter' not in para and
                  'Follow her on Twitter' not in para and
                  'You can read diverse opinions from our Board of Contributors' not in para and
                  'Read or Share this story:' not in para and
                  'RELATED COVERAGE' not in para and
                  'more:' not in para.lower() and
                  'MORE PHOTOS:' not in para and
                  'READ:' not in para and
                  'Find NJ.com' not in para and
                  'Follow him on Twitter' not in para and
                  'REUTERS/' not in para and
                  'via REUTERS' not in para and
                  'picture:' not in para.lower() and
                  'pictures:' not in para.lower() and
                  '(istockphoto)' not in para.lower() and
                  'documentary, physical challenges to Philly' not in para and
                  'pic.twitter.com' not in para and
                  'Staff Photographer' not in para and
                  'Flashpoint:' not in para and
                  'This story is about Published' not in para and  # ['tweet.user.screen_name']==dallasnews
                  'Share This Story On' not in para and  # ['tweet.user.screen_name']==dallasnews
                  'Staff Writer Contact' not in para and  # ['tweet.user.screen_name']==dallasnews
                  'Flashpoint:' not in para and  # ['tweet.user.screen_name']==dallasnews
                  'Related Image Expand / Collapse' not in para and  # ['tweet.user.screen_name']==FoxNews
                  'SCROLL DOWN FOR VIDEO' not in para and  # ['tweet.user.screen_name']==FoxNews
                  'call us direct on' not in para and  # ['tweet.user.screen_name']==TheSun
                  'READ ALSO:' not in para and
                  'RELATED:' not in para and
                  'RELATED CONTENT:' not in para and
                  'Related:' not in para and
                  'for breaking news alerts' not in para and
                  'is a Philly.com editor' not in para and
                  'please check your Spam or Junk folder' not in para and
                  'We pay for your stories' not in para and  # TheSun
                  'COLLECT IMAGE' not in para and  #
                  'see also' not in para.lower() and  #
                  'Times]' not in para and  #
                  'See:' not in para and  #
                  'SEE:' not in para and  # DID YOU SEE: MUST SEE:
                  'RELEVANT:' not in para and  #
                  '/ Chicago Tribune' not in para and  #
                  'follow him on facebook' not in para.lower() and  #
                  'follow her on facebook' not in para.lower() and  #
                  'The opinions expressed in this commentary' not in para and  #
                  'Story highlights' not in para and  #
                  'Where to Stream' not in para and  #
                  'More Options' not in para and  #
                  ('(AP)' in para or 'AP)' not in para) and  #
                  'Creators Syndicate' not in para and  #
                  'Advertisement Continue reading' not in para and  #
                  '/Flickr' not in para and  #
                  'Read the original' not in para and  #
                  'Your browser does not support the' not in para and  #
                  'SUPPORT THE GUARDIAN' not in para and  #
                  'please support us' not in para and  #
                  'Sign up and receive' not in para and  #
                  'file photo' not in para.lower() and  #
                  'Picture:' not in para and  #
                  'Pictures)' not in para and  #
                  'UPDATES:' not in para and  #
                  not re.match('UPDATE [\d]{0,1}:', para) and  #
                  'UPDATE:' not in para and  # controversial exclusion verify with experiments and analysis
                  'Corrections & Clarifications:' not in para and
                  'For complete coverage' not in para and
                  'click the link below' not in para and
                  'click here' not in para.lower() and
                  'spoiler:' not in para.lower() and
                  'RELATED CONTENT:' not in para and
                  'READ ON' not in para and
                  'Newsletter Sign Up Continue reading' not in para and
                  'Scripting must be enabled to view the information below' not in para and
                  'Keep up with this story and more by subscribing now' not in para and
                  'Andorra Angola Anguilla' not in para and
                  'Keep up to date with all our' not in para and
                  'read more' not in para.lower() and
                  'thanks for your continued support' not in para.lower() and
                  "we're thankful for your support" not in para.lower() and
                  'subscribe let our news meet your inbox' not in para.lower() and
                  'keep up with this story' not in para.lower() and
                  'to see all content on the sun' not in para.lower() and
                  'opens in new window' not in para.lower() and
                  'watch cbs' not in para.lower() and
                  'coverage:' not in para.lower() and
                  'briefing:' not in para.lower() and
                  'blog:' not in para.lower() and
                  'more to follow' not in para.lower() and
                  'for the latest news' not in para.lower() and
                  'like us on facebook' not in para.lower() and
                  'this is a breaking story' not in para.lower() and
                  'please refresh or return for updates' not in para.lower() and
                  'rate, review, share' not in para.lower() and
                  'subscribe & review on' not in para.lower() and
                  'share tweet pin email' not in para.lower() and
                  'originally published:' not in para.lower() and
                  'your browser does not support iframes' not in para.lower() and
                  'keep up to date with all our' not in para.lower() and
                  'this story will be updated' not in para.lower() and
                  'caption close image' not in para.lower() and
                  'view caption hide caption' not in para.lower() and
                  'subscribe let our news meet your inbox' not in para.lower() and
                  'new york daily news published' not in para.lower() and
                  'follow us on Twitter' not in para.lower() and
                  'subscribe to our channel' not in para.lower() and
                  'find us on Facebook' not in para.lower() and
                  'check our website' not in para.lower() and
                  u'veröffentlicht am' not in para.lower() and
                  'most read in tv' not in para.lower() and
                  'most trusted sources for news and information' not in para.lower() and
                  'visit our site:' not in para.lower() and
                  para.lower() != lead.lower() and
                  para.lower() != title.lower()
                  ]
    return '\n\n'.join(para_clean)


def remove_media(row):
    try:
        if 'tweet.retweeted_status.id' not in row or pd.isnull(row['tweet.retweeted_status.id']):
            tweet = row['tweet.text']
            if type(row['tweet.extended_tweet.entities.urls']) == list:
                urls = row['tweet.extended_tweet.entities.urls']
                hashtag = row['tweet.extended_tweet.entities.hashtags']
                symbol = row['tweet.extended_tweet.entities.symbols']
                media = row['tweet.extended_tweet.entities.media'] if type(
                    row['tweet.extended_tweet.entities.media']) == list else []
                usermention = row['tweet.extended_tweet.entities.user_mentions']
            else:
                urls = row['tweet.entities.urls']
                hashtag = row['tweet.entities.hashtags']
                symbol = row['tweet.entities.symbols']
                media = row['tweet.entities.media'] if type(row['tweet.entities.media']) == list else []
                usermention = row['tweet.entities.user_mentions']
        else:
            tweet = row['tweet.retweeted_status.text']
            if type(row['tweet.retweeted_status.extended_tweet.entities.urls']) == list:
                urls = row['tweet.retweeted_status.extended_tweet.entities.urls']
                hashtag = row['tweet.retweeted_status.extended_tweet.entities.hashtags']
                symbol = row['tweet.retweeted_status.extended_tweet.entities.symbols']
                media = row['tweet.retweeted_status.extended_tweet.entities.media'] if type(
                    row['tweet.retweeted_status.extended_tweet.entities.media']) == list else []
                usermention = row['tweet.retweeted_status.extended_tweet.entities.user_mentions']
            else:
                urls = row['tweet.retweeted_status.entities.urls']
                hashtag = row['tweet.retweeted_status.entities.hashtags']
                symbol = row['tweet.retweeted_status.entities.symbols']
                media = row['tweet.retweeted_status.entities.media'] if type(
                    row['tweet.retweeted_status.entities.media']) == list else []
                usermention = row['tweet.retweeted_status.entities.user_mentions']
        lst_entities = []
        for m in media + urls + usermention + symbol:
            mstart, mend = m['indices']
            lst_entities.append((mstart, mend))
        lst_entities.sort(key=lambda x: x[0])
        start_ent = [itm[0] for itm in lst_entities]
        end_ent = [itm[1] for itm in lst_entities]
        text_clean = []
        idx = 0
        while idx < len(tweet.strip()):
            if idx in start_ent:
                idx = end_ent[start_ent.index(idx)]
            else:
                text_clean.append(tweet[idx])
                idx += 1
    except Exception as e:
        print(e)
        print(row['tweet.id'])
        raise
    # return pd.Series({'tweet.clean_text': ' '.join(text_clean)})
    return ''.join(text_clean).strip(), tweet


def get_df_subset(row):
    username = row['tweet.user.name']
    screen_name = row['tweet.user.screen_name']
    url = row['text.url']
    tweet_id = row['tweet.id_str']
    headline = row['text.title']
    article_raw = row['text.text_b64']
    tweet_raw = row['tweet_raw']
    article = row['text.clean_article']
    tweet = row['tweet.clean_text']
    lead1 = row['text.meta_data.twitter.description'] if pd.notnull(row['text.meta_data.twitter.description']) else ''
    lead2 = row['text.meta_data.og.description'] if pd.notnull(row['text.meta_data.og.description']) else ''
    lead3 = row['text.meta_data.description'] if pd.notnull(row['text.meta_data.description']) else ''
    keywords1 = list(map(lambda x: x.strip(), re.split("[,;]+", row['text.meta_data.news_keywords']))) if pd.notnull(
        row['text.meta_data.news_keywords']) else []
    keywords2 = list(map(lambda x: x.strip(), re.split("[,;]+", row['text.meta_data.keywords']))) if pd.notnull(
        row['text.meta_data.keywords']) else []
    return pd.Series({'username': username,
                      'screen_name': screen_name,
                      'url': url,
                      'tweet_id': tweet_id,
                      'headline': headline,
                      'article': article,
                      'article_raw': article_raw,
                      'tweet': tweet,
                      'tweet_raw': tweet_raw,
                      'lead1': lead1,
                      'lead2': lead2,
                      'lead3': lead3,
                      'keywords1': keywords1,
                      'keywords2': keywords2,
                      })


non_bmp = re.compile(u"[^\U00000000-\U0000d7ff\U0000e000-\U0000ffff]", flags=re.UNICODE)
dotted = re.compile(u'[\u2026]+', re.UNICODE)

ucode_list = filter(non_bmp.findall, 'text')
for char in ucode_list:
    tweet = 'text'.replace(char, ' ')


def word_tknz_artsentence(row):
    sentences = row['article_sentences'].split('\n\n')
    proc_sent = []
    for sent in sentences:
        sent = _replace_html_entities(sent)
        sent = re.sub(non_bmp, ' ', sent)
        sent = re.sub(dotted, ' ', sent)
        sent = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', sent, flags=re.MULTILINE)
        sent = re.sub('\s+', ' ', sent).lstrip().rstrip()
        sent = ' '.join(nltk.word_tokenize(sent))
        proc_sent.append(sent)
    tknz_sent = list(map(lambda sent: ' '.join(nltk.word_tokenize(sent)), proc_sent))
    dgtz_sent = list(map(lambda sent: re.sub(r'\d', '%', sent), tknz_sent))
    return ' '.join(dgtz_sent)


def process_tweets(row):
    tweet_tkns = row['tweet'].split()
    twt_clean = [tkn for idx, tkn in enumerate(tweet_tkns)
                 if 'https' not in tkn
                 and 'RT' != tkn
                 and not (idx == 1 and '@' in tkn)
                 and not (idx == 0 and '@' in tkn)
                 ]
    twt_clean = twt_clean[:-1] if '@' in twt_clean[-1] else twt_clean
    tweet = ' '.join(twt_clean)
    tweet = _replace_html_entities(tweet)
    reduce_lengthening = re.compile(r"(.)\1{2,}")
    tweet = reduce_lengthening.sub(r"\1\1\1", tweet)
    remode_handles = re.compile(
        r"(?<![A-Za-z0-9_!@#\$%&*])@(([A-Za-z0-9_]){20}(?!@))|"
        r"(?<![A-Za-z0-9_!@#\$%&*])@(([A-Za-z0-9_]){1,19})(?![A-Za-z0-9_]*@)")
    tweet = remode_handles.sub(' ', tweet)
    tweet = re.sub(non_bmp, ' ', tweet)
    tweet = re.sub(dotted, ' ', tweet)
    tweet = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', tweet, flags=re.MULTILINE)
    tweet = re.sub('\s+', ' ', tweet).lstrip().rstrip()
    tweet = ' '.join(nltk.word_tokenize(tweet))
    tweet = re.sub(r'\d', '%', tweet)
    # return ' '.join(tweet_tknzr.tokenize(' '.join(twt_clean))), ' '.join(nltk.word_tokenize(' '.join(twt_clean)))
    return tweet


def process_headlines(row):
    headline = row['headline']
    headline = _replace_html_entities(headline)
    headline = re.sub('\s+', ' ', headline).lstrip().rstrip()
    headline = ' '.join(nltk.word_tokenize(headline))
    headline = re.sub(r'\d', '%', headline)
    return headline


def check_noisy_headline(row):
    if "print + digital deals, student, gift offers" in row['headline'].lower():
        return False
    if "the new yorker" in row['headline'].lower():
        return False
    if "janet christie" in row['headline'].lower():
        return False
    if "the right way to brag on instagram" in row['headline'].lower():
        return False
    if "hall of fame and shame" in row['headline'].lower():
        return False
    if "check out the full schedule" in row['headline'].lower():
        return False
    if "music highlights" in row['headline'].lower():
        return False
    if u'‘the daily’' in row['headline'].lower():
        return False
    if "live updates and how to watch" in row['headline'].lower():
        return False
    if "watch live" in row['headline'].lower():
        return False
    if "holiday photo is going viral for the best reason" in row['headline'].lower():
        return False
    if "chicago sun-times: chicago news, sports, politics, entertainment" in row['headline'].lower():
        return False
    if "today's top picks" in row['headline'].lower():
        return False
    if "starten sie ihre kampagne" in row['headline'].lower():
        return False
    if "free digital subscription" in row['headline'].lower():
        return False
    if "the week in photos" in row['headline'].lower():
        return False
    if "access to this page has been denied" in row['headline'].lower():
        return False
    if "newsgrid - al jazeera's interactive news hour" in row['headline'].lower():
        return False
    if "business calendar" in row['headline'].lower():
        return False
    if "this week on \"" in row['headline'].lower():
        return False
    if "tv info," in row['tweet'].lower():
        return False
    if "subscribe to" in row['tweet'].lower():
        return False
    if "added an editors" in row['tweet'].lower():
        return False
    if "evening news roundup" in row['tweet'].lower():
        return False
    if "101 east al jazeera" in row['username'].lower():
        return False
    # if "review :" in row['tokenized_headline']:
    #     return False
    if "i feel like..." in row['article'].lower():
        return False
    if "more comfortable online than out partying" in row['article'].lower():
        return False
    if "buy from amazon.com" in row['article'].lower():
        return False
    return True


def sent_tokenize_article(row):
    article = row['article']
    para_ascii = list(map(lambda sent: sent.strip(), article.strip().split('\n\n')))
    para_ = [para for para in para_ascii if len(para.strip().split()) > 2]
    para_clean = [para if (para[-1] == '.' or para[-1] == '?' or para[-1] == '!') else para + '.' for para in para_]
    article_clean = ' '.join(para_clean)
    sentences = nltk.sent_tokenize(article_clean)
    return '\n\n'.join(sentences)


def nonovlp_tokens(s_seq, l_seq, ovlp_tpl_lst, w_len, stem=False, return_wnd=False):
    s_tokens = get_ngrams(s_seq, 1, True, stem)  # s_seq = x['tokenized_tweet'].lower()
    s_tokens_set = set(s_tokens)
    sentences = sent_tokenize(l_seq)  # l_seq = x['tokenized_article'].lower()
    ovlp_idx, ovlp_ratio = ovlp_tpl_lst[0]
    win_len = min(w_len, len(sentences))
    wind_sent = [wsent for wsent in window(sentences, win_len)]
    assert len(wind_sent) == len(ovlp_tpl_lst), 'windowed sentences, %s, ovlp ratio %s' % (wind_sent, ovlp_tpl_lst)
    l_tokens = get_ngrams(' '.join(wind_sent[ovlp_idx]), 1, True, stem)
    l_tokens_set = set(l_tokens)
    intersect_ = s_tokens_set.intersection(l_tokens_set)
    if len(l_tokens_set) == 0 or len(intersect_) == 0:
        ovlp = 0.0
    else:
        ovlp = float(len(intersect_)) / len(s_tokens_set)
    assert ovlp == ovlp_ratio, '%s given ovlp, computed ovlp %s, %s' % (ovlp_ratio, ovlp, ovlp_tpl_lst)
    if return_wnd:
        return s_tokens_set - intersect_, ' '.join(wind_sent[ovlp_idx])
    else:
        return s_tokens_set - intersect_

re_np = re.compile('[A-Z]+\w*\s+')


def get_top_tokens(text, tkn_size, use_re=True):
    if use_re:
        tokens_ = ''.join(re_np.findall(text)).lower().split()
    else:
        tokens_ = text.lower().split()
    tkns_ = [tok for tok in tokens_ if tok not in stopwords_en and len(tok) > 2]
    tkns_stm = dict(Counter(list(map(stemmer.stem, tkns_))).most_common(tkn_size))
    return ' '.join(tkns_stm.keys())


def collect_leads(row):
    leads = []
    if row['lead1'] == row['lead2'] == row['lead3'] and len(row['lead1']) > 1:
        if not (row['headline'] in row['lead1'] or row['lead1'] in row['headline']):
            leads.append(row['lead1'])
    elif row['lead1'] == row['lead2'] and len(row['lead1'] + row['lead2']) > 1:
        if not (row['headline'] in row['lead1'] or row['lead1'] in row['headline']):
            leads.append(row['lead1'])
        if len(row['lead3']) > 1 and not (row['headline'] in row['lead3'] or row['lead3'] in row['headline']) \
                and not (row['lead1'] in row['lead3'] or row['lead3'] in row['lead1']):
            leads.append(row['lead3'])
    elif row['lead1'] == row['lead3'] and len(row['lead1'] + row['lead3']) > 1:
        if not (row['headline'] in row['lead1'] or row['lead1'] in row['headline']):
            leads.append(row['lead1'])
        if len(row['lead2']) > 1 and not (row['headline'] in row['lead2'] or row['lead2'] in row['headline']) \
                and not (row['lead1'] in row['lead2'] or row['lead2'] in row['lead1']):
            leads.append(row['lead2'])
    elif row['lead2'] == row['lead3'] and len(row['lead2'] + row['lead3']) > 1:
        if not (row['headline'] in row['lead2'] or row['lead2'] in row['headline']):
            leads.append(row['lead2'])
        if len(row['lead1']) > 1 and not (row['headline'] in row['lead1'] or row['lead1'] in row['headline']) \
                and not (row['lead2'] in row['lead1'] or row['lead1'] in row['lead2']):
            leads.append(row['lead1'])
    else:
        if len(row['lead1']) > 1 and not (row['headline'] in row['lead1'] or row['lead1'] in row['headline']):
            leads.append(row['lead1'])
        if len(row['lead2']) > 1 and not (row['headline'] in row['lead2'] or row['lead2'] in row['headline']):
            leads.append(row['lead1'])
        if len(row['lead3']) > 1 and not (row['headline'] in row['lead3'] or row['lead3'] in row['headline']):
            leads.append(row['lead1'])
    return leads


def tokenize_leads(row):
    leads = []
    for lead in row['all_leads']:
        lead = _replace_html_entities(lead)
        lead = re.sub('\s+', ' ', lead).lstrip().rstrip()
        lead = ' '.join(word_tokenize(lead))
        lead = re.sub(r'\d', '%', lead)
        leads.append(lead)
    return leads

flt_tokens = {'breaking', 'editorial', 'irish', 'perspective', 'solar', 'update', 'video', 'analysis', 'watchdog',
                  'live', 'timeline', 'review', 'watch', 'developing', 'most-read', 'readers', 'analysts', 'exclusive',
                  'opinion', '#', '.', 'ask', 'column', 'op-ed', 'lsu', 'university', 'commentary'
                                                                                      'robert', 'garrison', 'rory',
                  'michael', 'partick', 'david', 'john', 'jimmy', 'jp', 'chris', 'ap',
                  'noel', 'charleton', 'icymi', 'watch', 'more', 'in pics'}


def get_d2v_corpus(corpora):
    for i, line in enumerate(corpora):
        # For training data, add tags
        yield doc2vec.TaggedDocument(
            [tkn for tkn in line.encode('utf-8').decode('utf-8').lower().split() if tkn not in stopwords_en], [i])


def get_relevant_ratio_km(
        txt, cls, tokenize, vocab, c0_mdl, c1_mdl, c2_mdl, c3_mdl, c4_mdl, c5_mdl, c6_mdl, c7_mdl, tr_lmt=0.01):
    tkn_lst = [tk for tk in tokenize(txt) if tk in vocab]
    if len(tkn_lst) == 0.: return 0.
    else:
        if cls == 0:
            token_tr = np.array([c0_mdl[0, vocab[tkn]] for tkn in tkn_lst])
        elif cls == 1:
            token_tr = np.array([c1_mdl[0, vocab[tkn]] for tkn in tkn_lst])
        elif cls == 2:
            token_tr = np.array([c2_mdl[0, vocab[tkn]] for tkn in tkn_lst])
        elif cls == 3:
            token_tr = np.array([c3_mdl[0, vocab[tkn]] for tkn in tkn_lst])
        elif cls == 4:
            token_tr = np.array([c4_mdl[0, vocab[tkn]] for tkn in tkn_lst])
        elif cls == 5:
            token_tr = np.array([c5_mdl[0, vocab[tkn]] for tkn in tkn_lst])
        elif cls == 6:
            token_tr = np.array([c6_mdl[0, vocab[tkn]] for tkn in tkn_lst])
        elif cls == 7:
            token_tr = np.array([c7_mdl[0, vocab[tkn]] for tkn in tkn_lst])
        else:
            raise ValueError('No more than 8 classes')
        return sum(token_tr < tr_lmt) / float(len(tkn_lst))


def terms_relevance_score_km(txt, cls, tokenize, vocab, c0_mdl, c1_mdl, c2_mdl, c3_mdl, c4_mdl, c5_mdl, c6_mdl, c7_mdl):
    tkn_lst = [tk for tk in tokenize(txt) if tk in vocab]
    if len(tkn_lst) == 0.:
        return []
    else:
        if cls == 0:
            token_tr = [(tkn, c0_mdl[0, vocab[tkn]]) for tkn in tkn_lst]
        elif cls == 1:
            token_tr = [(tkn, c1_mdl[0, vocab[tkn]]) for tkn in tkn_lst]
        elif cls == 2:
            token_tr = [(tkn, c2_mdl[0, vocab[tkn]]) for tkn in tkn_lst]
        elif cls == 3:
            token_tr = [(tkn, c3_mdl[0, vocab[tkn]]) for tkn in tkn_lst]
        elif cls == 4:
            token_tr = [(tkn, c4_mdl[0, vocab[tkn]]) for tkn in tkn_lst]
        elif cls == 5:
            token_tr = [(tkn, c5_mdl[0, vocab[tkn]]) for tkn in tkn_lst]
        elif cls == 6:
            token_tr = [(tkn, c6_mdl[0, vocab[tkn]]) for tkn in tkn_lst]
        elif cls == 7:
            token_tr = [(tkn, c7_mdl[0, vocab[tkn]]) for tkn in tkn_lst]
        else:
            raise ValueError('No more than 8 classes')
        token_tr_srt = sorted(token_tr, key=lambda itm: itm[1])
        return token_tr_srt


def get_pareto_coverage_km_binwise(df, bins, km_class, pareto_dict):
    dict_nontr_inpareto = {}
    for tr in bins:
        df_tr = df[df.apply(lambda x: x['kmeans_class'] == km_class and len(np.array(
            dict(x['nonovlptkns_w5_trval_stw']).values())[np.array(dict(x['nonovlptkns_w5_trval_stw']).values())<tr]) > 0, axis=1)]
        df_zf = df_tr[df_tr.apply(
            lambda x:len([tk for tk in x['nonovlptkns_w5_trval_stw'] if tk in pareto_dict]) == len(
                x['nonovlptkns_w5_trval_stw']), axis=1)]
        dict_nontr_inpareto.update({tr: (df_tr.shape[0], float(df_zf.shape[0]) / df_tr.shape[0])})
    return sorted(dict_nontr_inpareto.items(), key=lambda itm: itm[0])


def colname_to_value(cvg_tpl_lst):
    tr_cvg_dict = {}
    for tpls in cvg_tpl_lst:
        tpls_lst = list(tpls)
        colname = tpls_lst[0]
        cvg_val = np.atleast_1d(tpls_lst[1])[-1]
        val = colname.replace('tr_per_twt_', '')
        cvg_tr = float(val)/(10**len(val))
        tr_cvg_dict.update({cvg_tr: cvg_val})
    return sorted(tr_cvg_dict.items(), key=lambda itm:itm[0], reverse=True)


def article_vector_tmp3(row, d2v_model):
    doc_tkns = [tkn for tkn in ' '.join(row['tokenized_article'].lower().split()[:200]).split() if
                tkn not in stopwords_en]
    inferred_vector = d2v_model.infer_vector(doc_tkns)
    return pd.Series({'artvec': inferred_vector})
