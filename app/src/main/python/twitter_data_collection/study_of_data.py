import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans
from MulticoreTSNE import MulticoreTSNE as TSNE
from gensim.models import doc2vec
from string import punctuation
from nltk.stem.snowball import SnowballStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
from .utils import process_articles, get_df_subset, remove_media, \
    check_noisy_headline, sent_tokenize_article, word_tknz_artsentence, process_tweets, process_headlines, collect_leads, \
    tokenize_leads, percentage_match_tokens, percentage_match_window_tokens, get_d2v_corpus, get_relevant_ratio_km, \
    terms_relevance_score_km, nonovlp_tokens, get_pareto_coverage_km_binwise, colname_to_value, get_top_tokens, \
    article_vector_tmp3

stop_words = stopwords.words('english') + list(punctuation)
stopwords_en = stopwords.words('english')
stemmer = SnowballStemmer("english")
token_pattern = re.compile(u'(?u)\\b\\w\\w+\\b')
tokenize_stm_stw = lambda doc: map(stemmer.stem, [w for w in token_pattern.findall(
    doc.lower()) if w not in stopwords_en and len(w) > 1])
stm_stw_corpus_tfidf = TfidfVectorizer(tokenizer=tokenize_stm_stw, smooth_idf=True)


def setup_argparser():
    my_parser = argparse.ArgumentParser(prog='twitter_read_stream')

    my_parser.add_argument('-v', '--version', action='version', version='Twitter data format transformer')
    my_parser.add_argument("-i",
                           "--input_file",
                           type=str,
                           help="input file containing data in a pickled pandas data frame",
                           default=None)
    my_parser.add_argument("-o",
                           "--output_folder",
                           type=str,
                           help="folder to write out to",
                           default=None)

    return my_parser


def save_doc2vec(data_df, path):
    df_artwise = data_df.drop_duplicates('tokenized_article')
    corpus_all = get_d2v_corpus(
        [' '.join(df_artwise.iloc[i]['tokenized_article'].split()[:100]) for i in range(df_artwise.shape[0])])
    train_corpus_all = list(corpus_all)
    d2v_all = doc2vec.Doc2Vec(dm=0, size=50, window=5, min_count=5, workers=70, epochs=8, hs=0, negative=15)
    d2v_all.build_vocab(train_corpus_all)
    d2v_all.train(train_corpus_all, total_examples=d2v_all.corpus_count, epochs=d2v_all.epochs)
    d2v_all.save(os.path.join(path, 'd2v.mdl'))


def get_naacl_data(data_path):
    df_tokenize_augoct = pd.read_pickle(os.path.join(data_path, 'raw_corpus/df_augoct_tknzd_b4ovlp.pkl'))
    df_tokenize_junaug = pd.read_pickle(os.path.join(data_path, 'raw_corpus/df_junaug_tknzd_b4ovlp.pkl'))
    df_tokenize_b4june = pd.read_pickle(os.path.join(data_path, 'raw_corpus/df_b4june_tknzd_b4ovlp.pkl'))
    df_tknz_data_octA8 = pd.read_pickle(os.path.join(data_path, 'raw_corpus/df_octbA8_tknzd_b4ovlp.pkl'))
    df_tknz_data_novA8 = pd.read_pickle(os.path.join(data_path, 'raw_corpus/df_novmA8_tknzd_b4ovlp.pkl'))
    df_tknz_data_decA8 = pd.read_pickle(os.path.join(data_path, 'raw_corpus/df_decmA8_tknzd_b4ovlp.pkl'))
    df_tknz_data_feb19 = pd.read_pickle(os.path.join(data_path, 'raw_corpus/df_feb19_tknzd_b4ovlp.pkl'))
    df_tknz_data_mar19 = pd.read_pickle(os.path.join(data_path, 'raw_corpus/df_mar19_tknzd_b4ovlp.pkl'))
    df_tknz_data_up2oct17 = pd.concat([df_tokenize_b4june, df_tokenize_junaug, df_tokenize_augoct], axis=0) #size 515543
    df_tknz_data_up2may18 = pd.concat([df_tknz_data_octA8, df_tknz_data_novA8, df_tknz_data_decA8], axis=0) #size 728403
    df_tknz_data_up2mar19 = pd.concat([df_tknz_data_feb19,df_tknz_data_mar19], axis=0) #size 297888
    df_tokenize = pd.concat([df_tknz_data_up2oct17,df_tknz_data_up2may18,df_tknz_data_up2mar19], axis=0) #size 1541834
    return df_tokenize


def get_cluster_tfidf(data_abs):
    df_d0 = data_abs[data_abs.kmeans_class == 1]
    df_d1 = data_abs[data_abs.kmeans_class == 2]
    df_d2 = data_abs[data_abs.kmeans_class == 3]
    df_d3 = data_abs[data_abs.kmeans_class == 4]
    df_d4 = data_abs[data_abs.kmeans_class == 5]
    df_d5 = data_abs[data_abs.kmeans_class == 6]
    df_d6 = data_abs[data_abs.kmeans_class == 7]
    df_d7 = data_abs[data_abs.kmeans_class == 8]
    cluster0_doc = ' '.join([df_d0.iloc[i]['tokenized_article'] for i in range(df_d0.shape[0])])
    cluster1_doc = ' '.join([df_d1.iloc[i]['tokenized_article'] for i in range(df_d1.shape[0])])
    cluster2_doc = ' '.join([df_d2.iloc[i]['tokenized_article'] for i in range(df_d2.shape[0])])
    cluster3_doc = ' '.join([df_d3.iloc[i]['tokenized_article'] for i in range(df_d3.shape[0])])
    cluster4_doc = ' '.join([df_d4.iloc[i]['tokenized_article'] for i in range(df_d4.shape[0])])
    cluster5_doc = ' '.join([df_d5.iloc[i]['tokenized_article'] for i in range(df_d5.shape[0])])
    cluster6_doc = ' '.join([df_d6.iloc[i]['tokenized_article'] for i in range(df_d6.shape[0])])
    cluster7_doc = ' '.join([df_d7.iloc[i]['tokenized_article'] for i in range(df_d7.shape[0])])
    tfidf_stm_stw_mdl_km = stm_stw_corpus_tfidf.fit([
        cluster0_doc, cluster1_doc, cluster2_doc, cluster3_doc, cluster4_doc, cluster5_doc, cluster6_doc, cluster7_doc])
    c0_mdl_v2 = tfidf_stm_stw_mdl_km.transform([cluster0_doc])
    c1_mdl_v2 = tfidf_stm_stw_mdl_km.transform([cluster1_doc])
    c2_mdl_v2 = tfidf_stm_stw_mdl_km.transform([cluster2_doc])
    c3_mdl_v2 = tfidf_stm_stw_mdl_km.transform([cluster3_doc])
    c4_mdl_v2 = tfidf_stm_stw_mdl_km.transform([cluster4_doc])
    c5_mdl_v2 = tfidf_stm_stw_mdl_km.transform([cluster5_doc])
    c6_mdl_v2 = tfidf_stm_stw_mdl_km.transform([cluster6_doc])
    c7_mdl_v2 = tfidf_stm_stw_mdl_km.transform([cluster7_doc])
    return c0_mdl_v2, c1_mdl_v2, c2_mdl_v2, c3_mdl_v2, c4_mdl_v2, c5_mdl_v2, c6_mdl_v2, c7_mdl_v2, \
           tfidf_stm_stw_mdl_km.vocabulary_


def plot_pareto_coverage(data_abs):
    df_d0 = data_abs[data_abs.kmeans_class == 0]
    df_d1 = data_abs[data_abs.kmeans_class == 1]
    df_d2 = data_abs[data_abs.kmeans_class == 2]
    df_d3 = data_abs[data_abs.kmeans_class == 3]
    df_d4 = data_abs[data_abs.kmeans_class == 4]
    df_d5 = data_abs[data_abs.kmeans_class == 5]
    df_d6 = data_abs[data_abs.kmeans_class == 6]
    df_d7 = data_abs[data_abs.kmeans_class == 7]
    cluster0_doc = ' '.join([df_d0.iloc[i]['tokenized_article'] for i in range(df_d0.shape[0])])
    cluster1_doc = ' '.join([df_d1.iloc[i]['tokenized_article'] for i in range(df_d1.shape[0])])
    cluster2_doc = ' '.join([df_d2.iloc[i]['tokenized_article'] for i in range(df_d2.shape[0])])
    cluster3_doc = ' '.join([df_d3.iloc[i]['tokenized_article'] for i in range(df_d3.shape[0])])
    cluster4_doc = ' '.join([df_d4.iloc[i]['tokenized_article'] for i in range(df_d4.shape[0])])
    cluster5_doc = ' '.join([df_d5.iloc[i]['tokenized_article'] for i in range(df_d5.shape[0])])
    cluster6_doc = ' '.join([df_d6.iloc[i]['tokenized_article'] for i in range(df_d6.shape[0])])
    cluster7_doc = ' '.join([df_d7.iloc[i]['tokenized_article'] for i in range(df_d7.shape[0])])
    c0_tkns_counter = Counter(tokenize_stm_stw(cluster0_doc))
    c1_tkns_counter = Counter(tokenize_stm_stw(cluster1_doc))
    c2_tkns_counter = Counter(tokenize_stm_stw(cluster2_doc))
    c3_tkns_counter = Counter(tokenize_stm_stw(cluster3_doc))
    c4_tkns_counter = Counter(tokenize_stm_stw(cluster4_doc))
    c5_tkns_counter = Counter(tokenize_stm_stw(cluster5_doc))
    c6_tkns_counter = Counter(tokenize_stm_stw(cluster6_doc))
    c7_tkns_counter = Counter(tokenize_stm_stw(cluster7_doc))
    float(sum(dict(c0_tkns_counter.most_common(2000)).values())) / sum(c0_tkns_counter.values())  # ~ 0.8 1800
    c0_tkns_dict = dict(c0_tkns_counter.most_common(2000))
    c1_tkns_dict = dict(c1_tkns_counter.most_common(2000))
    c2_tkns_dict = dict(c2_tkns_counter.most_common(2000))
    c3_tkns_dict = dict(c3_tkns_counter.most_common(2000))
    c4_tkns_dict = dict(c4_tkns_counter.most_common(2000))
    c5_tkns_dict = dict(c5_tkns_counter.most_common(2000))
    c6_tkns_dict = dict(c6_tkns_counter.most_common(2000))
    c7_tkns_dict = dict(c7_tkns_counter.most_common(2000))
    c0_tr_cvg_tpl = get_pareto_coverage_km_binwise(data_abs, [0.001, 0.005, 0.01, 0.05, 0.1, 1.], 1, c0_tkns_dict)
    c1_tr_cvg_tpl = get_pareto_coverage_km_binwise(data_abs, [0.001, 0.005, 0.01, 0.05, 0.1, 1.], 2, c1_tkns_dict)
    c2_tr_cvg_tpl = get_pareto_coverage_km_binwise(data_abs, [0.001, 0.005, 0.01, 0.05, 0.1, 1.], 3, c2_tkns_dict)
    c3_tr_cvg_tpl = get_pareto_coverage_km_binwise(data_abs, [0.001, 0.005, 0.01, 0.05, 0.1, 1.], 4, c3_tkns_dict)
    c4_tr_cvg_tpl = get_pareto_coverage_km_binwise(data_abs, [0.001, 0.005, 0.01, 0.05, 0.1, 1.], 5, c4_tkns_dict)
    c5_tr_cvg_tpl = get_pareto_coverage_km_binwise(data_abs, [0.001, 0.005, 0.01, 0.05, 0.1, 1.], 6, c5_tkns_dict)
    c6_tr_cvg_tpl = get_pareto_coverage_km_binwise(data_abs, [0.001, 0.005, 0.01, 0.05, 0.1, 1.], 7, c6_tkns_dict)
    c7_tr_cvg_tpl = get_pareto_coverage_km_binwise(data_abs, [0.001, 0.005, 0.01, 0.05, 0.1, 1.], 8, c7_tkns_dict)
    c0_tr_cvg = colname_to_value(c0_tr_cvg_tpl)
    c1_tr_cvg = colname_to_value(c1_tr_cvg_tpl)
    c2_tr_cvg = colname_to_value(c2_tr_cvg_tpl)
    c3_tr_cvg = colname_to_value(c3_tr_cvg_tpl)
    c4_tr_cvg = colname_to_value(c4_tr_cvg_tpl)
    c5_tr_cvg = colname_to_value(c5_tr_cvg_tpl)
    c6_tr_cvg = colname_to_value(c6_tr_cvg_tpl)
    c7_tr_cvg = colname_to_value(c7_tr_cvg_tpl)
    x = np.array([0, 1, 2, 3, 4, 5])
    plt.xticks(x, zip(*c0_tr_cvg)[0])
    plt.plot(x, zip(*c0_tr_cvg)[1], label='c0-cvg-dr')
    plt.plot(x, zip(*c1_tr_cvg)[1], label='c1-cvg-dr')
    plt.plot(x, zip(*c2_tr_cvg)[1], label='c2-cvg-dr')
    plt.plot(x, zip(*c3_tr_cvg)[1], label='c3-cvg-dr')
    plt.plot(x, zip(*c4_tr_cvg)[1], label='c4-cvg-dr')
    plt.plot(x, zip(*c5_tr_cvg)[1], label='c5-cvg-dr')
    plt.plot(x, zip(*c6_tr_cvg)[1], label='c6-cvg-dr')
    plt.plot(x, zip(*c7_tr_cvg)[1], label='c7-cvg-dr')
    plt.xlabel('threshold values')
    plt.ylabel('overlap-ratio of relevance-based to freq-based corpus')
    plt.legend(loc='lower left')
    plt.savefig('drelevance_vs_pareto_km8.png', bbox_inches="tight")


def plot_scut_article_overlap(df, save_path, file_name_tag):
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    import matplotlib.mlab as mlab
    df_abstw = df[df.apply(lambda x: x['ovlp_tw_wnd_top'] <= 0.7, axis=1)]
    df_abstw['stmovlp_tw_art'] = df_abstw.apply(
        lambda x: percentage_match_tokens(x['tokenized_tweet'].lower(), x['tokenized_article'].lower(), True), axis=1)
    df_abstw['stmovlp_hd_art'] = df_abstw.apply(
        lambda x: percentage_match_tokens(x['tokenized_headline'].lower(), x['tokenized_article'].lower(), True),
        axis=1)
    data1, data2 = df_abstw['stmovlp_tw_art'], df_abstw['stmovlp_hd_art']
    f, ax = plt.subplots()
    (mu, sigma) = norm.fit(data1)
    n, bins, patches = ax.hist(data1, 50, normed=True, alpha=0.75)
    y = mlab.normpdf(bins, mu, sigma)
    l = ax.plot(bins, y, 'r--', linewidth=2)
    ax.grid(True)
    f.savefig(os.path.join(save_path, 'histo_ugram_twart_%s.png'%(file_name_tag)), bbox_inches="tight")
    f, ax = plt.subplots()
    (mu, sigma) = norm.fit(data2)
    n, bins, patches = ax.hist(data2, 50, normed=True, alpha=0.75)
    y = mlab.normpdf(bins, mu, sigma)
    l = ax.plot(bins, y, 'r--', linewidth=2)
    ax.grid(True)
    f.savefig(os.path.join(save_path, 'histo_ugram_hdart_%s.png'%(file_name_tag)), bbox_inches="tight")


def create_dataset(df_tsr, save_path, file_name_tag, test_size=250, dev_size=250):
    df_tsr['art_top_tkns'] = df_tsr.apply(lambda x: get_top_tokens(x['tokenized_article'], 5), axis=1)
    df_tsr['twt_top_tkns'] = df_tsr.apply(lambda x: get_top_tokens(x['tokenized_tweet'], 5), axis=1)
    df_tsr_lead = df_tsr[df_tsr.apply(lambda x: len(
        [itm for itm in x['all_tokenized_leads'] if len(itm.strip().split()) > 5]) > 0, axis=1)]
    df_tsr_lead_hdl = df_tsr_lead[df_tsr_lead.apply(lambda x: len(x['tokenized_headline'].strip().split()) > 5, axis=1)]
    df_tsr_lead_hdl['mc_lead'] = df_tsr_lead_hdl.apply(
        lambda x: Counter([a for a in x['all_tokenized_leads'] if len(a.split()) > 5]).most_common(1)[0][0], axis=1)
    df_tsr_leq1 = df_tsr_lead_hdl.groupby('art_top_tkns').filter(lambda x: len(x) == 1).drop_duplicates(
        {'twt_top_tkns'}).drop_duplicates({'tokenized_headline'}).drop_duplicates({'lead1'})
    df_wts_c0 = df_tsr_leq1.apply(lambda x: x['kmeans_class'] == 0, axis=1)
    df_wts_c1 = df_tsr_leq1.apply(lambda x: x['kmeans_class'] == 1, axis=1)
    df_wts_c2 = df_tsr_leq1.apply(lambda x: x['kmeans_class'] == 2, axis=1)
    df_wts_c3 = df_tsr_leq1.apply(lambda x: x['kmeans_class'] == 3, axis=1)
    df_wts_c4 = df_tsr_leq1.apply(lambda x: x['kmeans_class'] == 4, axis=1)
    df_wts_c5 = df_tsr_leq1.apply(lambda x: x['kmeans_class'] == 5, axis=1)
    df_wts_c6 = df_tsr_leq1.apply(lambda x: x['kmeans_class'] == 6, axis=1)
    df_wts_c7 = df_tsr_leq1.apply(lambda x: x['kmeans_class'] == 7, axis=1)
    df_c0_smp = df_tsr_leq1.sample(
        n=test_size, weights=df_wts_c0, replace=False) if sum(df_wts_c0)>0 else pd.DataFrame(columns=df_tsr_leq1.columns)
    df_c1_smp = df_tsr_leq1.sample(
        n=test_size, weights=df_wts_c1, replace=False) if sum(df_wts_c1)>0 else pd.DataFrame(columns=df_tsr_leq1.columns)
    df_c2_smp = df_tsr_leq1.sample(
        n=test_size, weights=df_wts_c2, replace=False) if sum(df_wts_c2)>0 else pd.DataFrame(columns=df_tsr_leq1.columns)
    df_c3_smp = df_tsr_leq1.sample(
        n=test_size, weights=df_wts_c3, replace=False) if sum(df_wts_c3)>0 else pd.DataFrame(columns=df_tsr_leq1.columns)
    df_c4_smp = df_tsr_leq1.sample(
        n=test_size, weights=df_wts_c4, replace=False) if sum(df_wts_c4)>0 else pd.DataFrame(columns=df_tsr_leq1.columns)
    df_c5_smp = df_tsr_leq1.sample(
        n=test_size, weights=df_wts_c5, replace=False) if sum(df_wts_c5)>0 else pd.DataFrame(columns=df_tsr_leq1.columns)
    df_c6_smp = df_tsr_leq1.sample(
        n=test_size, weights=df_wts_c6, replace=False) if sum(df_wts_c6)>0 else pd.DataFrame(columns=df_tsr_leq1.columns)
    df_c7_smp = df_tsr_leq1.sample(
        n=test_size, weights=df_wts_c7, replace=False) if sum(df_wts_c7)>0 else pd.DataFrame(columns=df_tsr_leq1.columns)
    df_tsr_tst = pd.concat([df_c0_smp, df_c1_smp, df_c2_smp, df_c3_smp, df_c4_smp, df_c5_smp, df_c6_smp, df_c7_smp])
    df_tsr_leq1_remn = pd.concat([df_tsr_leq1, df_tsr_tst]).drop_duplicates({'art_top_tkns'}, keep=False)
    df_wts_c0 = df_tsr_leq1_remn.apply(lambda x: x['kmeans_class'] == 0, axis=1)
    df_wts_c1 = df_tsr_leq1_remn.apply(lambda x: x['kmeans_class'] == 1, axis=1)
    df_wts_c2 = df_tsr_leq1_remn.apply(lambda x: x['kmeans_class'] == 2, axis=1)
    df_wts_c3 = df_tsr_leq1_remn.apply(lambda x: x['kmeans_class'] == 3, axis=1)
    df_wts_c4 = df_tsr_leq1_remn.apply(lambda x: x['kmeans_class'] == 4, axis=1)
    df_wts_c5 = df_tsr_leq1_remn.apply(lambda x: x['kmeans_class'] == 5, axis=1)
    df_wts_c6 = df_tsr_leq1_remn.apply(lambda x: x['kmeans_class'] == 6, axis=1)
    df_wts_c7 = df_tsr_leq1_remn.apply(lambda x: x['kmeans_class'] == 7, axis=1)
    df_c0_smp2 = df_tsr_leq1_remn.sample(
        n=dev_size, weights=df_wts_c0, replace=False) if sum(df_wts_c0)>0 else pd.DataFrame(columns=df_tsr_tst.columns)
    df_c1_smp2 = df_tsr_leq1_remn.sample(
        n=dev_size, weights=df_wts_c1, replace=False) if sum(df_wts_c1)>0 else pd.DataFrame(columns=df_tsr_tst.columns)
    df_c2_smp2 = df_tsr_leq1_remn.sample(
        n=dev_size, weights=df_wts_c2, replace=False) if sum(df_wts_c2)>0 else pd.DataFrame(columns=df_tsr_tst.columns)
    df_c3_smp2 = df_tsr_leq1_remn.sample(
        n=dev_size, weights=df_wts_c3, replace=False) if sum(df_wts_c3)>0 else pd.DataFrame(columns=df_tsr_tst.columns)
    df_c4_smp2 = df_tsr_leq1_remn.sample(
        n=dev_size, weights=df_wts_c4, replace=False) if sum(df_wts_c4)>0 else pd.DataFrame(columns=df_tsr_tst.columns)
    df_c5_smp2 = df_tsr_leq1_remn.sample(
        n=dev_size, weights=df_wts_c5, replace=False) if sum(df_wts_c5)>0 else pd.DataFrame(columns=df_tsr_tst.columns)
    df_c6_smp2 = df_tsr_leq1_remn.sample(
        n=dev_size, weights=df_wts_c6, replace=False) if sum(df_wts_c6)>0 else pd.DataFrame(columns=df_tsr_tst.columns)
    df_c7_smp2 = df_tsr_leq1_remn.sample(
        n=dev_size, weights=df_wts_c7, replace=False) if sum(df_wts_c7)>0 else pd.DataFrame(columns=df_tsr_tst.columns)
    df_tsr_val = pd.concat([
        df_c0_smp2, df_c1_smp2, df_c2_smp2, df_c3_smp2, df_c4_smp2, df_c5_smp2, df_c6_smp2, df_c7_smp2])
    df_tsr_trn = pd.concat([df_tsr_lead_hdl, df_tsr_val, df_tsr_tst]).drop_duplicates({'tokenized_tweet'}, keep=False)
    print('train, test and val of sizes %s, %s, %s'%(df_tsr_trn.shape[0], df_tsr_tst.shape[0],df_tsr_val.shape[0]))
    df_tsr_trn.to_pickle(os.path.join(save_path, 'df_%s_tsr_trn.pkl'%file_name_tag))
    df_tsr_tst.to_pickle(os.path.join(save_path, 'df_%s_tsr_tst.pkl'%file_name_tag))
    df_tsr_val.to_pickle(os.path.join(save_path, 'df_%s_tsr_val.pkl'%file_name_tag))
    if not os.path.exists(os.path.join(save_path, 'corpus_baseline')):
        os.mkdir(os.path.join(save_path, 'corpus_baseline'))
    df_tsr_trn[['tokenized_article', 'tokenized_tweet', 'tokenized_headline', 'mc_lead', 'kmeans_class']].to_json(
        os.path.join(save_path, 'corpus_baseline/%s_train.json'%file_name_tag), orient='records', lines=True)
    df_tsr_tst[['tokenized_article', 'tokenized_tweet', 'tokenized_headline', 'mc_lead', 'kmeans_class']].to_json(
        os.path.join(save_path, 'corpus_baseline/%s_test.json'%file_name_tag), orient='records', lines=True)
    df_tsr_val[['tokenized_article', 'tokenized_tweet', 'tokenized_headline', 'mc_lead', 'kmeans_class']].to_json(
        os.path.join(save_path, 'corpus_baseline/%s_valid.json%file_name_tag'), orient='records', lines=True)


def create_term_relevance_corpus(df_tsr_trn, save_path, file_name_tag, sample_size=40000):
    df_trn_tr_lt23ge00 = df_tsr_trn[(df_tsr_trn.tr_per_twt_0005 < 0.23)]
    df_trn_tr_lt33ge23 = df_tsr_trn[(df_tsr_trn.tr_per_twt_0005 >= 0.23) & (df_tsr_trn.tr_per_twt_0005 < 0.33)]
    df_trn_tr_lt41ge33 = df_tsr_trn[(df_tsr_trn.tr_per_twt_0005 >= 0.33) & (df_tsr_trn.tr_per_twt_0005 < 0.41)]
    df_trn_tr_lt51ge41 = df_tsr_trn[(df_tsr_trn.tr_per_twt_0005 >= 0.41) & (df_tsr_trn.tr_per_twt_0005 < 0.51)]
    df_trn_tr_lt99ge51 = df_tsr_trn[(df_tsr_trn.tr_per_twt_0005 >= 0.51)]
    df_trn_tr_lt23ge00_smp = df_trn_tr_lt23ge00.sample(n=sample_size, replace=False)
    df_trn_tr_lt33ge23_smp = df_trn_tr_lt33ge23.sample(n=sample_size, replace=False)
    df_trn_tr_lt41ge33_smp = df_trn_tr_lt41ge33.sample(n=sample_size, replace=False)
    df_trn_tr_lt51ge41_smp = df_trn_tr_lt51ge41.sample(n=sample_size, replace=False)
    df_trn_tr_lt99ge51_smp = df_trn_tr_lt99ge51.sample(n=sample_size, replace=False)
    if not os.path.exists(os.path.join(save_path, 'corpus_tr_lt23')):
        os.mkdir(os.path.join(save_path, 'corpus_tr_lt23'))
    if not os.path.exists(os.path.join(save_path, 'corpus_tr_lt33')):
        os.mkdir(os.path.join(save_path, 'corpus_tr_lt33'))
    if not os.path.exists(os.path.join(save_path, 'corpus_tr_lt41')):
        os.mkdir(os.path.join(save_path, 'corpus_tr_lt41'))
    if not os.path.exists(os.path.join(save_path, 'corpus_tr_lt51')):
        os.mkdir(os.path.join(save_path, 'corpus_tr_lt51'))
    if not os.path.exists(os.path.join(save_path, 'corpus_tr_lt99')):
        os.mkdir(os.path.join(save_path, 'corpus_tr_lt99'))
    df_trn_tr_lt23ge00_smp[['tokenized_article', 'tokenized_tweet', 'kmeans_class']].to_json(
        os.path.join(save_path, 'corpus_tr_lt23/%s_train.json'%file_name_tag), orient='records', lines=True)
    df_trn_tr_lt33ge23_smp[['tokenized_article', 'tokenized_tweet', 'kmeans_class']].to_json(
        os.path.join(save_path, 'corpus_tr_lt33/%s_train.json'%file_name_tag), orient='records', lines=True)
    df_trn_tr_lt41ge33_smp[['tokenized_article', 'tokenized_tweet', 'kmeans_class']].to_json(
        os.path.join(save_path, 'corpus_tr_lt41/%s_train.json'%file_name_tag), orient='records', lines=True)
    df_trn_tr_lt51ge41_smp[['tokenized_article', 'tokenized_tweet', 'kmeans_class']].to_json(
        os.path.join(save_path, 'corpus_tr_lt51/%s_train.json'%file_name_tag), orient='records', lines=True)
    df_trn_tr_lt99ge51_smp[['tokenized_article', 'tokenized_tweet', 'kmeans_class']].to_json(
        os.path.join(save_path, 'corpus_tr_lt99/%s_train.json'%file_name_tag), orient='records', lines=True)


def main():
    parser = setup_argparser()
    args = parser.parse_args()
    resource_path = args.output_folder
    if not os.path.exists(resource_path):
        os.mkdir(resource_path)
    flag_ = 'apr20' # tag will used for storing files

    #region clean data
    data = pd.read_pickle(args.input_file)
    data['text.clean_article'] = data.apply(process_articles, axis=1)
    data[['tweet.clean_text', 'tweet_raw']] = data.apply(lambda x: remove_media(x), axis=1, result_type="expand")
    df_sub = data.apply(get_df_subset, axis=1)
    df_uniq = df_sub.drop_duplicates(subset='tweet', keep='last')
    df_uniq = df_uniq[df_uniq.apply(check_noisy_headline, axis=1)]
    #endregion

    #region tokenize
    df_tokenize = df_uniq[df_uniq.apply(
        lambda x: len(x['article'].split()) > 99 and len(x['tweet'].split()) > 4, axis=1)]
    df_tokenize['article_sentences'] = df_tokenize.apply(sent_tokenize_article, axis=1)
    df_tokenize['tokenized_article'] = df_tokenize.apply(word_tknz_artsentence, axis=1)
    df_tokenize['tokenized_tweet'] = df_tokenize.apply(process_tweets, axis=1)
    df_tokenize['tokenized_headline'] = df_tokenize.apply(process_headlines, axis=1)
    #endregion

    # region load clean and tokenized NAACL 19 dataset
    # df_tokenize = get_naacl_data(resource_path)
    # endregion

    # region preprocessing
    df_tknz = df_tokenize.drop_duplicates('tokenized_tweet')
    df_org = df_tknz.groupby('username').filter(lambda x: len(x) > 100)
    df_tknz_teaser = df_org.drop_duplicates('tokenized_tweet')
    df_tsr_neqhd = df_tknz_teaser[df_tknz_teaser.apply(
        lambda x: x['tokenized_headline'].lower() != x['tokenized_tweet'].lower() and
                  x['tokenized_tweet'].lower() not in x['tokenized_headline'].lower(), axis=1)]
    df_tsr_nohdemb = df_tsr_neqhd[df_tsr_neqhd.apply(
        lambda x: x['tokenized_headline'].lower() not in x['tokenized_tweet'].lower(), axis=1)]
    df_tsr_ldsent = df_tsr_nohdemb[df_tsr_nohdemb.apply(lambda x: sum(
        [x['tokenized_tweet'].lower() in sent or sent in x['tokenized_tweet'].lower() for sent in
         sent_tokenize(x['tokenized_article'].lower())]) == 0, axis=1)]
    data_scut = df_tsr_ldsent[df_tsr_ldsent.apply(
        lambda x: len([x for x in x['tokenized_tweet'].strip().split() if x not in stopwords_en]) > 5, axis=1)]
    data_scut['all_leads'] = data_scut.apply(lambda x: collect_leads(x), axis=1)
    data_scut['all_tokenized_leads'] = data_scut.apply(lambda x: tokenize_leads(x), axis=1)
    data_scut['stmovlp_tw_lds'] = data_scut.apply(lambda x: percentage_match_tokens(
        x['tokenized_tweet'].lower(), ' '.join(x['all_tokenized_leads']).lower(), True), axis=1)
    data_scut['stmovlp_tw_wnd5'] = data_scut.apply(lambda x: percentage_match_window_tokens(
        x['tokenized_tweet'].lower(), x['tokenized_article'].lower(), 5, True), axis=1)
    data_scut['ovlp_tw_wnd_top'] = data_scut.apply(
        lambda x: max(x['stmovlp_tw_wnd5'][0][1], x['stmovlp_tw_lds']), axis=1)
    #endregion

    # region cluster docs
    save_doc2vec(data_scut, resource_path)
    d2v = doc2vec.Doc2Vec.load(os.path.join(resource_path, 'd2v.mdl'))
    df_d2v = data_scut.apply(lambda x: article_vector_tmp3(x, d2v), axis=1)
    np_d2v = np.array([df_d2v.iloc[i]['artvec'] for i in range(df_d2v.shape[0])])
    tsne_d2v = TSNE(n_jobs=70, perplexity=100).fit_transform(np_d2v)
    df_tsne = pd.DataFrame(tsne_d2v, index=df_d2v.index, columns=['tsne_x', 'tsne_y'])
    df_d2v_wtsne = pd.concat([df_d2v, df_tsne], axis=1)
    kmeans_d2v = KMeans(8, init='k-means++', max_iter=10000, n_jobs=10).fit(np_d2v)
    df_kmn = pd.DataFrame(kmeans_d2v.labels_, index=df_d2v.index, columns=['kmeans_class'])
    df_d2v_wtsne_wkmn = pd.concat([df_d2v_wtsne, df_kmn], axis=1)
    data_kmn8 = pd.concat([data_scut, df_d2v_wtsne_wkmn], axis=1)
    # endregion

    # region overlap filter
    data_kmn8['ovlp_tw_wnd_top'] = data_kmn8.apply(
        lambda x: max(x['stmovlp_tw_wnd5'][0][1], x['stmovlp_tw_lds']), axis=1)
    data_neqhd = data_kmn8[data_kmn8.apply(
        lambda x: x['tokenized_headline'].lower() != x['tokenized_tweet'].lower() and
                  x['tokenized_tweet'].lower() not in x['tokenized_headline'].lower(), axis=1)]
    data_ldsent = data_neqhd[data_neqhd.apply(lambda x: sum(
        [x['tokenized_tweet'].lower() in sent or sent in x['tokenized_tweet'].lower() for sent in
         sent_tokenize(x['tokenized_article'].lower())]) == 0, axis=1)]
    data_abs = data_ldsent[data_ldsent.apply(lambda x: 0.2 <= x['ovlp_tw_wnd_top'] <= 0.8, axis=1)]
    # endregion

    # region plotting
    plot_scut_article_overlap(data_scut, resource_path, flag_)
    # plot_pareto_coverage(data_abs)
    # endregion

    #region term relevance filter
    c0_mdl, c1_mdl, c2_mdl, c3_mdl, c4_mdl, c5_mdl, c6_mdl, c7_mdl, vocabulary_ = get_cluster_tfidf(data_abs)
    data_abs['tr_per_twt_0005'] = data_abs.apply(lambda x: get_relevant_ratio_km(
        x['tokenized_tweet'], x['kmeans_class'], tokenize_stm_stw, vocabulary_,
        c0_mdl, c1_mdl, c2_mdl, c3_mdl, c4_mdl, c5_mdl, c6_mdl, c7_mdl, 0.005), axis=1)

    data_abs['twt_tr_scores_stw'] = data_abs.apply(lambda x: terms_relevance_score_km(
        x['tokenized_tweet'], x['kmeans_class'], tokenize_stm_stw, vocabulary_,
        c0_mdl, c1_mdl, c2_mdl, c3_mdl, c4_mdl, c5_mdl, c6_mdl, c7_mdl), axis=1)
    data_abs['nonovlp_tw_tkn_w5'] = data_abs.apply(lambda x: nonovlp_tokens(
        x['tokenized_tweet'].lower(), x['tokenized_article'].lower(), x['stmovlp_tw_wnd5'], 5, True, False), axis=1)
    data_abs['nonovlptkns_w5_trval_stw'] = data_abs.apply(
        lambda x: {s: v for s, v in dict(x['twt_tr_scores_stw']).items() if
                   s in [itm[0] for itm in list(x['nonovlp_tw_tkn_w5'])]}, axis=1)
    data_abs['ovlptkns_w5_trval_stw'] = data_abs.apply(
        lambda x: {s: v for s, v in dict(x['twt_tr_scores_stw']).items() if
                   s not in [itm[0] for itm in list(x['nonovlp_tw_tkn_w5'])]}, axis=1)
    df_tsr = data_abs[data_abs.apply(lambda x: len(x['nonovlptkns_w5_trval_stw']) > 1 and len(
        np.array(list(dict(x['nonovlptkns_w5_trval_stw']).values()))[
            (np.array(list(dict(x['nonovlptkns_w5_trval_stw']).values())) < 0.005) & (
                    np.array(list(dict(x['nonovlptkns_w5_trval_stw']).values())) > 0.0001)]) > 0, axis=1)]
    #endregion

    df_tsr.to_pickle(os.path.join(resource_path, 'df_%s_tsr_win5_2_8_wnvlplt0005.pkl'%flag_))
    create_dataset(df_tsr, resource_path, file_name_tag=flag_)
    df_tsr_trn = pd.read_pickle(os.path.join(resource_path, 'df_%s_tsr_trn.pkl'%flag_))
    create_term_relevance_corpus(df_tsr_trn, resource_path, flag_)


if __name__ == '__main__':
    main()

