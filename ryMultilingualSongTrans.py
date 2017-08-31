'''

ryMultilingualSongTrans001.py
ryKaraokBattleMachine003.py

多線程式在本作品中初步得到精進的機會。

多語歌聲轉寫，多年前(2003左右至今至少13年了吧)，
即有王重凱的碩士論文起頭，這些年來一直還保有於餘溫，
今年6月，還參加張智星的學生之碩士論文口試，還承接此一主題，
似乎喚起塵封的記憶。

現在以Python的基礎之下，
重新來過，似乎比較游刃有餘。
這幾年來，斷斷續續增加了日文的能力。
又站在 Google 巨人的肩膀之上，
多語言(zh-TW, ja, en)之語音辨識引擎似乎穩固，
就不用自己發展了吧！

也許仍須發展的引擎是台語，那是Google目前沒做的吧。

本程式特別依賴 Vpython， 它讓 3D 成為簡單的可能。
又找到2個重要的核心演算法模組，

一個是 speech_recognition, 
協助了 透過雲端向 Google Speech API 請求語音辨認引擎。

另一個則有關 pitch detection 的演算法，
2個參考來源都要查一下。 

Renyuan Lyu

2016/08/02

'''

import pyaudio
import wave

import threading
import copy
import time

import numpy as np

from ryF0Estimate import freq_from_autocorr

from visual import *

import speech_recognition as sr


scene= display(title= '''
    Pitch detection and Lyric Transcription 
    in Multilanguages with 3D Scenery, 
    by Renyuan Lyu''')

scene.autoscale= False # 場景不要自動調整大小。

class Ry音類:
    def __init__(self):
    
        self.音= pyaudio.PyAudio()

        self.樣本格式= pyaudio.paInt16
        self.樣本寬=   self.音.get_sample_size(self.樣本格式)

        self.通道數= 1     # 道
        self.取樣率= 16000 # 點/秒

        self.框長=   256   # 點/框
        self.總秒數= 16    # 秒
        
        self.流= self.音.open(
                    format=   self.樣本格式, 
                    channels= self.通道數,
                    rate=     self.取樣率, 
                    frames_per_buffer= self.框長,
                    input=    True, 
                    output=   True)

        self.框數=  int(self.取樣率 *self.總秒數 /self.框長 )
        self.錄音框們=  [[] for i in range(self.框數)]
        
        
        
        self.錄音框已滿= False
        self.初能量mean= 0 
        self.初能量std= 0 
        self.現能量= 0
        self.tF1= 0        # 錄音線程之迴圈執行時間
        
        self.地板= self.Z0= -50
        
        self.有語音嗎= False
        
        self.錄音中= False
        self.放音中= False
        self.f1_能量中= False
        self.f4_基頻中= False
        
        # 開始囉....
        #self.main()
        
        self.錄音線= threading.Thread(target= self.錄音線程)
        self.放音線= threading.Thread(target= self.放音線程)
        self.能量線= threading.Thread(target= self.f1_能量)
        self.基頻線= threading.Thread(target= self.f4_基頻)
        self.波形線= threading.Thread(target= self.f0_波形)
        
        self.語音辨認線= threading.Thread(target= self.f6_語音辨認) 
        
        self.特徵線= threading.Thread(target= self.f00_特徵) 
        self.有音偵測線= threading.Thread(target= self.f01_有音偵測) 
        

        self.start()

    def start(self):
              
        self.錄音線.start()
        #self.放音線.start()
        
        self.能量線.start()
        
        #self.有音偵測線.start()
        


        self.基頻線.start()
        
        ####self.波形線.start()
        
        self.語音辨認線.start()
        
        #self.特徵線.start()
 
        
        
        
        print('Enjoy it ....')        

    def 錄音線程(self):
        #global 錄音框們, 錄音中, i現框, 錄音框已滿
        
        print('錄音線開始 ....')
        self.i現框= 0
        self.錄音框已滿= False

        self.錄音中= True
        
        while (self.錄音中==True) and (self.錄音框已滿== False):
            框= self.流.read(self.框長)
            self.錄音框們[self.i現框%self.框數]= 框
            self.i現框 += 1
            if self.i現框>= self.框數: self.錄音框已滿= True
        print('.... 錄音框已滿 ....')
        
        while (self.錄音中==True):
            框= self.流.read(self.框長)
            self.錄音框們[self.i現框%self.框數]= 框
            self.i現框 += 1
        print('.... 錄音線結束')
     
    def 放音線程(self):
        #global 錄音框們, i現框 , 放音中, 放音框們
        
        print('放音線開始 ....')
        
        print('等待錄音框已滿 ....')
        while self.錄音框已滿==False: time.sleep(.01)
        
        self.放音中= True
        while self.放音中==True:
            框= self.錄音框們[(self.i現框-1)%self.框數]
            self.流.write(框)
        
        print('.... 放音線結束')        

    def f00_特徵(self):
    
        標= label()
        
        while self.錄音框已滿==False: time.sleep(.01)
        
        self.總秒數,
        self.錄音框們,
        
        self.i現框,
        self.框數,
        self.i現框%self.框數,
        
        
        self.tF1= 0
        
        self.錄音框已滿,
        self.錄音中,
        
        self.有語音嗎= False
        self.現能量= 0.0
        self.初能量mean= 0.0
        self.初能量std= 1.0
        self.基頻= 0
        self.文= ''

        
        self.f00_特徵中= True
        while self.f00_特徵中:
            特徵= '''
                a00= {a00}\na01= {a01}\na02= {a02}\na03= {a03}\n
                a04= {a04}\na05= {a05}\na06= {a06}\na07= {a07}\n
                a08= {a08}\na09= {a09}\na10= {a10}\na11= {a11}\n
                a12= {a12}\n'''.format(
                a00= self.總秒數,
                #self.錄音框們,
                
                a01= self.i現框,
                a02= self.框數,
                a03= self.i現框%self.框數,
                
                a04= self.tF1,
                
                a05= self.錄音框已滿,
                a06= self.錄音中,
                
                a07= self.有語音嗎,
                a08= self.現能量,
                a09= self.初能量mean,
                a10= self.初能量std,
                a11= self.基頻,
                a12= self.文,

                )
            
            標.text= self.特徵= 特徵
            標.pos= (-10,10,-10)
            rate(100)
        
    def f01_有音偵測(self):
        
        標= label()
        while self.錄音框已滿==False: time.sleep(.01)
        
        self.能量們=       np.array([])
        self.能量超過門檻= None
        
        i1= i0= di= 0
        self.有音之框們= []
        self.xL= [None for i in range(10)]
        self.yL= [None for i in range(10)]
        
        self.uttNum= 0
        self.f01_有音偵測中= True
        while self.f01_有音偵測中:
            標.text= '{}'.format(self.i現框)

            #xL= self.能量超過門檻 #= self.能量們 > (self.初能量mean + self.初能量std*3)
            
            if self.現能量 >  (self.初能量mean + self.初能量std*2):
                i1= i0= self.i現框
                di= 0
                while self.現能量 >  (self.初能量mean + self.初能量std*2):
                    #time.sleep(.01)
                    di= self.i現框 - i0
                if di > 10:
                    i1= self.i現框
                    
                    self.有音之框們 += [(i0,  i1,  di)]
                    
                    # 從錄音框中，撈出幾個音框，存起來。
                    i0 -= 2
                    i1 += 2
                    if (i0)%self.框數 < (i1)%self.框數:
                        x= b''.join(self.錄音框們[(i0)%self.框數:(i1)%self.框數]) 
                    else:
                        x= b''.join( self.錄音框們[(i0)%self.框數:]
                                    +self.錄音框們[0:(i1)%self.框數]) 
            
                    self.xL[self.uttNum%len(self.xL)]= x
                    
                    #y= np.fromstring(x, dtype= np.int16)
                    #self.yL[uttNum%len(self.yL)]= y
                    self.uttNum += 1
                    
                    
                    
            標.text= '{}, {} 段 有聲音'.format(self.i現框, len(self.有音之框們))
            
            rate(100)
    
    def f0_波形(self):
        #global tF1, f1中, R0, Z0, 初能量mean, 初能量std, 現能量
        
        
        T= self.總秒數
        R= 10
        Z=   1
        Z0= vector(0, self.地板, 0) #self.Z0 #0
        
        print('f0 開始 ....')

        
        灰色= (.5,.5,.5)
        #o= sphere(color= 灰色, make_trail= True, interval= T-1, retain=100)
        o1= o= curve(color= color.red, pos= Z0)
        o2= curve(color= color.green,  pos= Z0)


        while self.錄音框已滿==False: time.sleep(.01)
        

        self.f0_波形中= True
        t=   0
        dt= .01
        while self.f0_波形中==True:
            
            #
            # 從錄音框撈出1框來處理，平均每點之絕對值
            #
            x= self.錄音框們[(self.i現框-1)%self.框數]
            
            y= np.fromstring(x, dtype= np.int16)
            
            y1= y # 波形
            
            y2= np.fft.fft(y1) # 頻譜
            y2= np.abs(y2)
            
            y1 = y1[0::10]    # downsample to draw, 不然點數太多會有延遲。
            y1 = y1*.001      # scaling to draw
            
            y2 = y2[0:len(y2)//2:10]  # 頻率範圍僅一半
            y2 = y2*.0001      # scaling to draw
            
            t1L= np.arange(len(y1))
            o1L= t1L*0
            
            t2L= np.arange(len(y2)) *2 *10 # 空間範圍可加倍，
            o2L= t2L*0
            
            z1= list(zip(t1L, o1L, y1))
            z2= list(zip(o2L, t2L, y2))

            z1= [vector(z)+Z0 for z in z1]
            z2= [vector(z)+Z0 for z in z2]
            
            #
            # 用 Vpython 來顯示
            #
            o1.pos= z1
            o2.pos= z2

            
            rate(1/dt)
            o1.pos= []
            o2.pos= []
        
    def f1_能量(self):
        #global tF1, f1中, R0, Z0, 初能量mean, 初能量std, 現能量
        
        
        T= self.總秒數
        R= 10
        Z=   1
        Z0= vector(0, self.地板, -10) #self.Z0 #0
        
        print('f1 開始 ....')
        b= box(opacity= .5, material= materials.earth)
        for u in [(1,0,0),(0,1,0),(0,0,1)]:
            v= vector(u)
            a= arrow(axis= v, color= v)
        
        灰色= (.5,.5,.5)
        o= sphere(color= 灰色, make_trail= True, interval= T-1, retain=100)
        o.pos= Z0 # (0,self.地板,0)


        while self.錄音框已滿==False: time.sleep(.01)
        
        # 算一下　平均能量及標準差
        i0= self.i現框
        
        zL= []
        for k in range(self.框數):
            
            x= self.錄音框們[(i0-k)%self.框數]
            y= np.fromstring(x, dtype= np.int16)
            
            z= sum(abs(y))/self.框長
            if z>0: z= np.log(z)
            
            zL += [z]
        
        
        
        zL= np.array(zL)
        
        self.能量們= zL
        
        self.初能量mean= zL.mean()
        self.初能量std=  zL.std()

        self.f1_能量中= f1中= True
        
        self.現能量= self.初能量mean  
        self.tF1= t= 0  # 螢幕上作圖橫坐標(時間，用此變數同步其他 線程)
        dt= .01
        
        
        
        while self.f1_能量中==True:
            
            #
            # 從錄音框撈出1框來處理，平均每點之絕對值
            #
            x= self.錄音框們[(self.i現框-1)%self.框數]
            
            y= np.fromstring(x, dtype= np.int16)
            z= np.abs(y).mean() #sum(abs(y))/框長
            
            if z>0: z= np.log(z) # 取對數能量，但避開 log(負數)

            self.能量們[(self.i現框-1)%self.框數]= self.現能量= z
            

            #
            # 用 Vpython 來顯示，圓周運動。
            #
            色相= (self.i現框-1)%self.框數/ self.框數 #(t%T)/T
            
            o.pos= vector(R*cos(2*pi*色相),R*sin(2*pi*色相), z*Z) + Z0
            
            
            o.color= color.hsv_to_rgb((色相,1,1))
            
            
            
                    
            self.tF1= t # 為了傳出去。同步其他 線程
            t += dt
            
            rate(1/dt)
        
    def pitchQuantization(self, f):
    
        noteName= [
             'A ', 'A#', 'B ', 'B ', 'C#', 'D ','D#','E ','F ','F#','G ','G#',
             'a ', 'a#', 'b ', 'c ', 'c#', 'd ','d#','e ','f ','f#','g ','g#'
             ]
        
        f= max(1,f)
        
        n= int(round(12*np.log2(f/440)))
        
        F= 440 * (2**(n/12))
        
        n += 12
        return F, noteName[n%len(noteName)]
        
    def f4_基頻(self):
        #global f4中, R0, R
        #global 現能量, 初能量mean, 初能量std, tF1
            
        R= 10
        Z= .2
        Z0= self.Z0 #-50
        
        maxF0= 4000
        
        T= self.總秒數
        
        oF0= o= sphere(color= color.white, #yellow, #magenta, 
                  make_trail= True, 
                  #trail_type="points", 
                  trail_type= "points",
                  interval= T-1, 
                  
                  retain= 50)
                  
        oF0.trail_object.size= 10
        
        o能量= sphere(color= color.red, #magenta, 
                  make_trail= True, 
                  trail_type="curve", 
                  interval= T-1, 
                  retain= 50)
        標= label(opacity= .1)
        
        音符名串標=     label(opacity= .1)
        音符名串標.pos= (0, self.地板+100,10)

        # 畫 幾條水平線 當作音高參考，
        # 要知道 pitch in semitone (S) 
        # 以及 pitch in Hz (H)之對數關係。
        # 大概是 S= log2(H), 細節要再查一下，
        #
        # S= log2(H/440) * 12
        # H= 440 時， S=   0 == 'note_A'，鋼琴中間附近的 'la'
        # H= 880 時， S=  12 == 'note_a'，高八度的 'la'
        # H= 220 時， S= -12 == 'note_A,'，低八度的 'la'
        #
        # 然後用 色相 = 0 .... 1 均勻分配 12 平均律 的 每個 semitone
        #
        
        #for yy in np.linspace(Z0,50,10):
        
        譜線刻度= np.logspace(-1,4,num=12*5, base=2)*100/2**4+Z0
        
        #np.ceil(譜線刻度*10)
        '''
        array([   32.,    34.,    36.,    38.,    40.,    42.,    45.,    48.,
          50.,    54.,    57.,    60.,    64.,    68.,    72.,    76.,
          80.,    85.,    90.,    96.,   102.,   108.,   114.,   121.,
         128.,   136.,   144.,   153.,   162.,   172.,   183.,   194.,
         205.,   218.,   231.,   245.,   259.,   275.,   292.,   309.,
         328.,   348.,   369.,   391.,   415.,   440.,   466.,   495.,
         525.,   556.,   590.,   626.,   663.,   703.,   746.,   791.,
         839.,   890.,   943.,  1000.])
        '''
        
        for nn, yy in enumerate(譜線刻度):

            譜線= curve(pos= [(Z0 ,yy, 0), (Z0+100,yy,0)])
            譜線.color= color.hsv_to_rgb((nn%12/12,1,1))
        
        #
        # 再畫幾條垂直線，當作時間線。用灰色畫就好，比較不會喧賓奪主。
        #
        縱線數= 16
        縱線刻度= np.linspace(Z0, Z0+100, 縱線數+1)
        
        for nn, xx in enumerate(縱線刻度):

            縱線= curve(pos= [(xx ,2**(-1)*100/2**4+Z0, 0), (xx,2**4*100/2**4+Z0,0)])
            灰色= vector(1,1,1)*(1-(nn%4)/4)
            縱線.color= 灰色        
        
        #
        # 等待錄音線程填滿第一圈，其他動作才開始。
        #
        while self.錄音框已滿==False: time.sleep(.01)
        
        self.f4_基頻中= f4中= True
        
        dt= .01
        
        分析框數= 3
        hamm= np.hamming(self.框長*分析框數)
        
        音符名串= ''
        該對音符取樣了= True
        
        while self.f4_基頻中==True:
            
            # 從錄音框中，撈出幾個音框，求出基頻。
            x= b''.join([self.錄音框們[(self.i現框-j)%self.框數] 
                        for j in range(分析框數,0,-1)])
            
            y= np.fromstring(x, dtype= np.int16)
            
            #y *=  hamm
            y= y * hamm
            
            z= 0
            try:
                z= freq_from_autocorr(y, self.取樣率)
            except:
                pass
            
            #
            # 用 統計學 高斯分布，3倍標準差以上的機率已經低於 .15% 來做語音判定
            #
            有語音嗎= ( self.現能量 > (self.初能量mean + self.初能量std*3))
            
            z= min(z, maxF0)
            
            z, 音符名= self.pitchQuantization(z) # 量化至最近的 12 平均律 之頻率
            
            z= z * 有語音嗎
            
            self.有語音嗎= 有語音嗎
            self.基頻= z
            
            #print('z= ',z)
            
       
            t= self.tF1  # 為了同步 f1_能量
            
            #o.pos= vector(R*cos(2*pi/T*t),R*sin(2*pi/T*t),z*Z+Z_0)
            
            色相= (self.i現框-1)%self.框數/ self.框數 #(t%T)/T
            
            o.pos= vector((色相-1/2)*100,     z*Z+Z0,     0 )    #int(t/T)*10)
            
            
            
            if self.有語音嗎==True:
                標.visible= True
                標.pos= o.pos
                標.text= '{} {}'.format(int(z), 音符名) # 看整數即可                
            else:
                標.visible= False
                #o.trail_object.visible= False
            
            o能量.pos= vector((色相-1/2)*100,  Z0, 有語音嗎*10)  #int(t/T)*10)
            
            
            #色相= (t%T)/T
            #色相= (self.i現框-1)%self.框數/ self.框數 #(t%T)/T
            
            o.color= color.hsv_to_rgb((色相,1,1))
            o能量.color= color.hsv_to_rgb((色相,1,1))
         
            if ((self.i現框-1) % self.框數 == 0): 
                音符名串= ''
                該對音符取樣了= True
            
            #
            # 何時該對音符取樣似乎與節拍器有關，現在每2條縱線間隔為1秒，
            # 因此， %框數//縱線數//2==0 將代表每秒 2 拍，每分鐘 120 拍。
            #
            #
            if  ((self.i現框-1)%(self.框數//縱線數//2)==0) and 該對音符取樣了==False: 
                該對音符取樣了= True
            
            if 該對音符取樣了==True:
                if (self.有語音嗎==True):   音符名串 += 音符名
                else:                       音符名串 += '. '
                #print('音符名串= ', 音符名串) # print 會偷吃時間
                該對音符取樣了= False
            
            音符名串標.text= 音符名串
            #音符名串標.pos= o能量.pos
            
            
            
            
            
            if z >1024: # 口哨基頻通常大於1000，此為了處理口哨
                o.trail_object.color= color.yellow
            else:
                o.trail_object.color= color.white
            #tF2= t # 為了傳出去。
            #t += dt
            rate(1/dt)
            
    def f6_語音辨認(self, lang= 'ja'):
        '''
        可使用鍵盤 'j', 'e', 't'，來切換語言，
        'j' == 'ja', 'japan'
        'e' == 'en', 'english'
        't' == 'zh-TW', "traditional Chinese"
        '''
        
        辨=  sr.Recognizer()
        
        Z0= (0, self.地板+100+20,20)
        
        標0= 標= label(pos=Z0, height=30) # 現框字型最大
        
        Z1= (0, self.地板+100+30,30)
        標1=     label(pos=Z1, height=20) # 前1框字型
        
        Z2= (0, self.地板+100+40,40)
        標2=     label(pos=Z2, height=15) # 前2框字型
        
        #
        #安全措施，靜候第一個錄音迴圈已滿才開始做事。
        #
        while self.錄音框已滿==False: time.sleep(.01)
        
        self.f6_語音辨認中= True
        
        N= self.框數
        M= N  # 從現框回推 總共取 M 框來辨識
        
        lang= 'ja'#'zh-TW' # 預設值

        i前框= self.i現框-M+1 
        
        self.文= ''
        
        while self.f6_語音辨認中==True:
        
            #
            # 檢查改變語言的按鍵，決定辨認標的的語言
            #
            # 每次都取 N 框去辨識，
            # 除非按下[空白]或[Enter]鍵來提前中斷
            # 這其實是人工取語音斷點之意思。
            #
            key= ''

            while (self.i現框)%N != 0:
                if scene.kb.keys:
                    key= scene.kb.getkey()  # 取得鍵盤按鍵
                   
                if key in ['\n', ' ']: # Enter 或 空白鍵
                    break      # 提前醒來，跳出睡覺迴圈
                    
                elif key in ['1','j']:
                    lang= 'ja'
                elif key in ['2','e']:
                    lang= 'en'
                elif key in ['3','t']:
                    lang= 'zh-TW'
                else:
                    pass
                
                time.sleep(.01)
            
            #
            # 決定要送交語音辨識引擎的語音片段，這裡最需小心處理
            #
            # x= getSpeechForAsrEngin()
            #
            
            delta框= self.i現框 - i前框
            M= min(delta框, N-1)
            #
            # copy speech at i現框 = 0
            #
            
            i0= (self.i現框-1-M)%N # 語音起點在 M 點以前
            i1= (self.i現框-1)%N   # 語音終點就是現在的前1框
            if i0 < i1:
                x= b''.join(self.錄音框們[i0:i1+1]) 
            else: # i1<i0
                x= b''.join(self.錄音框們[i0:] + self.錄音框們[0:i1+1])

            i前框= self.i現框  # 此刻終點，為下刻起點。
            
            #'''
            # 以上會抓整個錄音框，太長了！ 目前好像設為 10 sec
            # 最好是做一下靜音偵測，
            
            #
            # 若 現框 i0 能量 小 且 找到能量 小 之前框位置 i1
            # i1 ... i0
            #???????
            #??????? 再想想！ 2016/07/31
            #
            '''
            N= self.框數
            
            y= self.錄音框們[(self.i現框+1)%N: N] + self.錄音框們[0:(self.i現框-1)%N] 
            x= b''.join(y) 
            '''
            #
            # recognize for i現框增加 (勿超過框數) 
            #
            音= sr.AudioData(x, 
                        self.取樣率, #source.SAMPLE_RATE, 
                        self.樣本寬)  #source.SAMPLE_WIDTH)
            #
            # 加一下鍵盤控制，切換語言
            #
            '''
            if scene.kb.keys:
                key= scene.kb.getkey()
                if    key=='1': lang= 'ja'
                elif  key=='2': lang= 'en'
                elif  key=='3': lang= 'zh-TW'
                else: pass
            '''
            
            if   lang=='ja':
                標.color= color.yellow                
            elif lang=='en':
                標.color= color.red
            else: # lang= 'zh-TW'
                標.color= color.cyan
            

                    
            #self.文= ''
            try:
                if   lang=='ja':
                    文= 辨.recognize_google(音, language='ja')                    
                elif lang=='en':
                    文= 辨.recognize_google(音, language='en')                    
                else: # lang= 'zh-TW'
                    文= 辨.recognize_google(音, language='zh-TW')
                    
                self.文= '{} ({})'.format(文, lang)
                
            except:
                pass
            
            if 標.text != self.文:
                標2.text= 標1.text
                標1.text= 標.text
            
            
            print('self.文= ', self.文)
            標.text= self.文
            
            ##rate(1/5) # 
        

        pass
    
    def 停止(self):
    
        self.錄音中= False
        self.放音中= False
        self.f1_能量中= False
        self.f4_基頻中= False
        
       
        #'''
        self.流.stop_stream()
        self.流.close()
        self.音.terminate()
        #'''
     
if __name__=='__main__':
    
    ry音= Ry音類()
    
