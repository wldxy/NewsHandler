import LDA
import pandas as pd
import MySQLdb as mdb
import collections
import datetime as dt
import requests
import numpy as np

def connect():
    info = {
        "host": "10.60.42.202",
        "user": "root",
        "passwd": "GCers+518",
        "db": "news",
        "charset": "utf8"
    }
    conn = mdb.connect(**info)
    return conn

def getDocs(d):
    docs = list()
    def solve(df):
        #print df["content"]
        if df["content"] == None:
            docs.append("")
        else:
            docs.append(df["content"])
    d.apply(solve, axis=1)
    return docs

def singleDayHandler(df):
    r = collections.defaultdict(int)
    value = df["result"]
    for i in value:
        for item in i:
            r[item[0]] += item[1]
    return pd.Series({"lda": r})

def printDF(df):
    print df

conn = connect()
sql = "select * from news"
news = pd.read_sql_query(sql, conn)
# mask = ((d['time'] > datetime.datetime(2015, 1, 1)) & (d['time'] < datetime.datetime(2015, 12, 30)))
# d = d[mask]

docs = getDocs(news)

lda = LDA.LDA()
lda.segment(docs)

print "start training model"
lda.ldatrain()

print "training model finish"

news["result"] = lda.corpus_lda
gt = news[["time", "result"]].groupby(pd.Grouper(freq='1D', key='time'))

ans = gt.apply(singleDayHandler)
# for i in range(20):
#     print lda.model.print_topic(i)

future = "IF1706"
url = "http://stock2.finance.sina.com.cn/futures/api/json.php/CffexFuturesService.getCffexFuturesDailyKLine?symbol="
price = requests.get(url+future)
price = price.json()
time = list()
p = list()
for i in price:
    time.append(dt.datetime.strptime(i[0], "%Y-%m-%d"))
    p.append(float(i[1]))
price = pd.DataFrame({"time": time, "price": p})
ans["time"] = ans.index
total = pd.merge(price, ans, on="time")

def getTopic(df):
    for i in range(20):
        if df["lda"].has_key(i):
            df[str(i)] = df["lda"][i]
        else:
            df[str(i)] = 0
    return df
data = total.apply(getTopic, axis=1)
data = data.drop(["lda", "time"], axis=1)
result = np.corrcoef(data, rowvar=0)*0.5+0.5
r0 = result[0]
dd = dict()
for i in range(20):
    # print i, r0[i+1]
    dd[i] = r0[i+1]
# print dd
topic_sort = sorted(dd.items(), key=lambda d: d[1], reverse=True)
topic_choose = list()
for i in range(3):
    topic_choose.append(topic_sort[i][0])

news_list = list()
def choose_news(df):
    global topic_choose
    global news_list
    topic = df["result"]
    for i in topic_choose:
        if i in [x for x, y in topic]:
            news_list.append(df["url"])
news.apply(choose_news, axis=1)
print news_list
