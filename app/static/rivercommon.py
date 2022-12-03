import os
import os.path
from datetime import datetime
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from pathlib import Path

class WatherLegend():
    def __init__(self):
        self.legend = {}
        self.bitmasknames = []

    def __eq__(self, other) -> bool:
        if not isinstance(other, WatherLegend):
            raise TypeError("Операнды должны иметь тип WatherLegend!")
        selfkeysset = set(self.legend.keys())
        otherkeysset = set(other.legend.keys())
        if selfkeysset == otherkeysset:
            return True
        return False


    def createlegend(self, rawdata = ''):
        soup = BeautifulSoup(rawdata, "html.parser")
        alltables = soup.find_all('table', class_='myTable')
        numb = 0
        for tbl in alltables:
            if tbl.find('tbody') is not None:
                legendstr = tbl.tbody.find_all('tr')
                for onestr in legendstr:
                    elems = onestr.find_all('td')
                    if len(elems) == 2:
                        self.legend[elems[0].text] = {'numb' : numb, 'desc' : elems[1].text}
                        self.bitmasknames.append(elems[0].text)
                        numb += 1

    def to_html(self):
        outtext = '<table>'
        for key, val in self.legend.items():
            desc = val['desc']
            outtext += f'<tr><td>{key}</td><td>{desc}</td></tr>'
        outtext += '</table>'
        return outtext
    
    def createbitmask(self, flags = []):
        bitmask = [False for x in range(len(self.legend))]
        for flag in flags:
            if flag in self.legend.keys():
                bitmask[self.legend[flag]['numb']] = True
        return bitmask

    def createflags(self, bitmask = []):
        flags = []
        for curnum, curbit in enumerate(bitmask):
            if curbit:
                for key, val in self.legend.items():
                    if val['numb'] == curnum:
                        flags.append(key)
        return flags

class WatherPost():
    def __init__(self):
        self.postcode = ''
        self.year = ''
        self.name = ''

    def loadfromtable(self, table=''):
        if len(table) == 0:
            return False
        allinfo = table.find_all('p')
        for oneelem in allinfo:
            if 'id' in oneelem.attrs.keys():
                if oneelem['id'] == 'kod_hpr':
                    self.postcode = oneelem.text
                if oneelem['id'] == 'year':
                    self.year = oneelem.text
                if oneelem['id'] == 'river_post':
                    self.name = oneelem.text
        return True

class RiverInfo():
    def __init__(self, year = 1000, legend = WatherLegend()):
        self.year = year
        self.legend = legend
        self.datadescr = ['Дата','Уровень'] + legend.bitmasknames
        self.data = pd.DataFrame(data=None, columns=self.datadescr)
        self.data['Дата'] = self.data['Дата'].astype('datetime64')
        self.data = self.data.set_index('Дата')
        self.data['Уровень'] = self.data['Уровень'].astype('int')
        for i in legend.bitmasknames:
            self.data[i] = self.data[i].astype('bool')
        #self.riverdata = pd.DataFrame

    def parseflagsstring(self, curstr = ''):
        outinfo={'level': -1, 'flags': []}
        lvl = np.nan#-1
        flgs = []
        rawstr = curstr.strip()
        elems = rawstr.split()
        if len(elems) > 0:
            elnew = ''
            if elems[0].isdigit():
                lvl = int(elems[0])
            else:
                elnew = rawstr
            if len(elems) == 2:
                elnew = elems[1]
            for i in ['прмз', 'прсх']:
                if elnew.find(i) > -1:
                    flgs.append(i)
                    elnew = elnew.partition(i)[0] + elnew.partition(i)[2]
            for i in list(elnew):
                flgs.append(i)
        outinfo['level'] = lvl
        outinfo['flags'] = flgs
        return outinfo

    def loadfromtable(self, table=''):
        if len(table) == 0:
            return False
        
        alltr = table.find_all('tr')
        for onetr in alltr:
            alltd = onetr.find_all('td')
            if len(alltd) == 13 and len(onetr.text) > 2:
                curday = 0
                for curnum, onetd in enumerate(alltd):
                    if curnum == 0:
                        curday = onetd.text
                    else:
                        curmonth = curnum
                        info = self.parseflagsstring(onetd.text)
                        curdate = f'{curday}/{curmonth}/{self.year}'
                        try:
                            datetime.strptime(curdate, "%d/%m/%Y")
                        except:
                            pass
                        else:
                            lvl = info['level']
                            flgs = info['flags']
                            bitflags = self.legend.createbitmask(flgs)
                            infolist = [lvl] + bitflags
                            self.data.loc[datetime(int(self.year), int(curmonth), int(curday))] = infolist
        self.data = self.data.sort_index(ascending=True)

    def loadfromfile(self, fname=''):
        if not os.path.exists(fname):
            return False
        self.data = pd.read_csv(fname, index_col='Дата',  sep=';', encoding='utf-8', parse_dates=['Дата'])
        #print(self.data.columns)
        #print(self.data.index.dtype)
        #print(self.data.head())

                        
class DataFile():
    def __init__(self, fname='', postinfodir='postsinfo'):
        self.legend = WatherLegend()
        self.fname = fname
        self.postinfodir = postinfodir
        self.year = 0
        self.postinfo = []
        self.riverinfo = []

    def loadfile(self):
        if not os.path.exists(self.fname):
            return False
        self.rawdata = ''
        with open(self.fname, 'r', encoding='utf-8') as mf:
            self.rawdata = mf.read()
        self.legend.createlegend(self.rawdata)
        return True
    
    def getriverlevel(self):
        soup = BeautifulSoup(self.rawdata, "html.parser")
        alltables = soup.find_all('table')
        year = 0
        for tbl in alltables:
            if 'class' in tbl.attrs.keys():
                if 'table' in tbl['class']:
                    curpost = WatherPost()
                    curpost.loadfromtable(tbl)
                    self.postinfo.append(curpost)
                    year = curpost.year
                if 'calend' in tbl['class']:
                    if str(tbl.text).find('Число') == 1:
                        print(self.postinfo[-1].postcode, '     ', self.postinfo[-1].name)
                        curriverinfo = RiverInfo(year, self.legend)
                        curriverinfo.loadfromtable(tbl)
                        self.riverinfo.append(curriverinfo)

    def savepostsinfo(self):
        assert len(self.postinfo) == len(self.riverinfo), 'Количество постов не совпадает с количеством данных об уровнях'
        for i in range(len(self.postinfo)):
            fname = self.postinfo[i].postcode + '.csv'
            curfname = Path(self.postinfodir, fname)
            if os.path.exists(curfname):
                curinfo = RiverInfo()
                curinfo.loadfromfile(curfname)
                #self.riverinfo[i].data = pd.concat([curinfo.data.iloc[curinfo.data.index != self.riverinfo[i].data.index], self.riverinfo[i].data], axis=0)
                self.riverinfo[i].data = pd.concat([curinfo.data, self.riverinfo[i].data], axis=0)
                self.riverinfo[i].data = self.riverinfo[i].data.sort_index(ascending=True)

            self.riverinfo[i].data.to_csv(curfname, sep=';', encoding='utf-8')

    def loadpost(self, fname = ''):
        if not os.path.exists(fname):
            return False
        curriverinfo = RiverInfo()

class InitData():
    def __init__(self):
        bases = ['Подкаменная Тунгуска ', 'Нижняя Тунгуска ']
        for curyear in range(2008, 2018, 1):
            for curbase in bases:
                curfname = curbase + str(curyear) + '.xls'
                curfname = Path('rawinfo', curfname)
                print(curfname)
                md = DataFile(fname=curfname)
                md.loadfile()
                md.getriverlevel()
                md.savepostsinfo() 

class ReportHTML():
    def __init__(self, fname='report'):
        self.fname = fname
        self.lines = [f'<html><body><h1><center>Данные точки наблюдения с кодом {fname}</center></h1><br>']

    def addsubheader(self, stext):
        self.lines.append(f'<h2>{stext}</h2><br>')
    
    def addpic(self, picname, width = 50):
        self.lines.append(f'<br><br><center><img src="{picname}" width="{width}%"></center><br>')

    def addtext(self, curtext):
        self.lines.append(f'{curtext}<br><br>')

    def savereport(self):
        self.lines.append('</body></html>')
        with open(self.fname + '.html', 'w', encoding='utf-8') as repfile:
            repfile.writelines(self.lines)
