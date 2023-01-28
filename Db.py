import sqlite3

class Db(object):
    def open(self, path, create):
        print("open db...")
        self.conn = sqlite3.connect(path)
        print("...Opened database successfully")

        if create:
            self.conn.execute('''CREATE TABLE IF NOT EXISTS a (
 id INTEGER PRIMARY KEY     AUTOINCREMENT,
 a           TEXT    NOT NULL,
 b           TEXT    NOT NULL
);''')


    
    def insert(self, arr1, arr2):
        strOfArr1 = ";".join([str(iv) for iv in arr1])
        strOfArr2 = ";".join([str(iv) for iv in arr2])
        print(strOfArr1)
        self.conn.execute("INSERT INTO a (a,b) VALUES ('"+strOfArr1+"','"+strOfArr2+"')")
        self.conn.commit()



    # returns array of [stimulus, output, gradientStrength]
    def queryById(self, id):
        cursor = self.conn.execute("SELECT id, a, b from a where id="+str(id))
        
        aStr=None
        bStr=None
        for row in cursor:
            #print ("id="+ str(row[0]))
            #print ("a="+ row[1])
            #print ("b="+ row[2])
            aStr=row[1]
            bStr=row[2]
        aArr=[float(iv) for iv in aStr.split(';')]
        bArr=[float(iv) for iv in bStr.split(';')]
        
        return [aArr,bArr, 1.0]
    
    def close(self):
        self.conn.close()

    #conn.close()

# sink to send training data into
class Sink(object):
    def __init__(self):
        self.db = Db()
        self.db.open(True)
    
    def append(self, arr):
        self.db.insert(arr[0], arr[1])

    def close(self):
        self.db.close()
