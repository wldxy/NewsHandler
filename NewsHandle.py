import LDA
import pandas as pd
import MySQLdb as mdb

def connect():
    info = {
    "host": "localhost",
    "user": "root",
    "passwd": "163613",
    "db": "news",
    "charset": "utf8"
    }
    conn = mdb.connect(**info)
    return conn

def getDocs(d):
    docs = list()
    def solve(df):
        #print df["content"]
        docs.append(df["content"])
    d.apply(solve, axis=1)
    return docs

def singleDayHandler(df):
    r = list()
    #for

conn = connect()
sql = "select * from news limit 100"
d = pd.read_sql_query(sql, conn)
docs = getDocs(d)

lda = LDA.LDA()
lda.segment(docs)
lda.ldatrain()
print lda.result

d["result"] = lda.result
gt = d[["time", "result"]].groupby(pd.Grouper(freq='1D', key='time'))

# lda = LDA.LDA