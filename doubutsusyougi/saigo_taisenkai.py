
import socket
import time
import copy
import random
import jouhou

BUFSIZE = 1024
serverName = "localhost"
#serverName = "10.2.72.166"
serverPort = 4444

s = socket.socket(socket.AF_INET,
socket.SOCK_STREAM)
s.connect((serverName, serverPort))

msg = s.recv(BUFSIZE).rstrip().decode()
print("msg ; ",msg)

global b

line = "0"

if (msg == "You are Player1."):
    player = "1"
    aiteplayer = "2"
# ↓ 念のためif文
elif (msg == "You are Player2."):
    player = "2"
    aiteplayer ="1"

def makemasuToKoma():
    s.send(("board" + "\n").encode())
    time.sleep(0.1)
    boardinfo = s.recv(BUFSIZE).rstrip().decode()

    masuToKoma = {}
    for mk in boardinfo.split(","):
        mk = mk.strip()
        #if mk != "" and mk.split(" ")[1] != "--":
        if mk != "":
            masuToKoma[mk.split(" ")[0]] = mk.split(" ")[1]
    return(masuToKoma)

def masuToXy(str):
    # 引数：マスを表す文字列（例：'B2'）
    # 返り値：座標を表すタプル（例: (1,1)）
    if str[0] == "A":
        x = 0
    elif str[0] == "B":
        x = 1
    elif str[0] == "C":
        x = 2
     #持ち駒
    elif ((str[0] == "D") or (str[0] == "E")):
        x = 8

    global y
        
    if str[1] == "1":
        y = 0
    elif str[1] == "2":
        y = 1
    elif str[1] == "3":
        y = 2
    elif str[1] == "4":
        y = 3
    
    return (x,y)

def xyToMasu(t):
    # 引数：座標を表すタプル（例: (1,1)）
    # 返り値：マスを表す文字列（例：'B2'）
    if t[0] == 0:
        xstr = "A"
    elif t[0] == 1:
        xstr = "B"
    elif t[0] == 2:
        xstr = "C"
       
    if t[1] == 0:
        ystr = "1"
    elif t[1] == 1:
        ystr = "2"
    elif t[1] == 2:
        ystr = "3"
    elif t[1] == 3:
        ystr = "4"

    return xstr + ystr

def tyakusyukanou(masuToKoma, player):
    lionKiki = ((0,  -1), (1,  -1), (-1, -1), (1,  0), (-1,  0), (1,  1),  (-1,  1), (0,   1))
    #lionTe = []

    hiyokoKiki = ((0, -1), (9, 9))
    #hiyokoTe = []

    niwatoriKiki = ((0,  1), (-1,  0), (1,  0), (-1,  -1),  (1,  -1), (0, -1))
    #niwatoriTe = []

    elephantKiki = ((1,  -1), (-1, -1),  (1,  1),  (-1,  1))
    #elephantTe = []

    giraffeKiki = ((0,  -1), (1,  0), (-1,  0), (0,  1))
    #giraffeTe = []

    if(player == "2"):
        hiyokoKiki = ((0, 1), (9, 9))
        niwatoriKiki = ((0,  -1), (1,  0), (-1,  0), (1,  1),  (-1,  1), (0,   1))

    # 持ち駒を同じように表現、B2を基準にして、全てのマス＝12マス動けることにする
    motigomaKiki = ((0, 0), (0,  -1), (1,  -1), (-1, -1), (1,  0), (-1,  0), (1,  1),  (-1,  1), (0,   1), (-1, 2), (0, 2), (1, 2))
    #motigomaTe = []

    #着手可能手全てを入れる
    #kanouTe = []
    tyouhuku = 1

    for srcStr, komaStr in masuToKoma.items():
        # ライオンを見つけた
        if komaStr[0] == "l" and komaStr[1] == player:
            # ライオンのマスを座標に変換
            (srcX, srcY) = masuToXy(srcStr)

            # ライオンの動き方すべてについて
            for (deltaX, deltaY) in lionKiki:

                # 行先のマスを計算
                dstX = srcX + deltaX
                dstY = srcY + deltaY

                # 行先が盤内の場合のみ考慮
                if dstX >= 0 and dstX <= 2 and dstY >= 0 and dstY <= 3:
                    # 行先をマスを表す文字列に変換
                    dstStr = xyToMasu((dstX, dstY))
                    
                    # 盤面のディクショナリを見て、
                    # 行先に駒がないか、あるいは自分の駒ではないとき着手可能
                    if masuToKoma.get(dstStr) is None or masuToKoma.get(dstStr)[1] != player:
                        yield (srcStr, dstStr)
                        #kanouTe.append((srcStr, dstStr))
                        # lionTe.append("mv " + srcStr + " " + dstStr)
        
        # ひよこを見つけた
        elif komaStr[0] == "c" and komaStr[1] == player:
            # ひよこのマスを座標に変換
            (srcX, srcY) = masuToXy(srcStr)
            # print(srcStr)
            # print(srcX, srcY)

            # ひよこの動き方すべてについて
            for (deltaX, deltaY) in hiyokoKiki:

                # 行先のマスを計算
                dstX = srcX + deltaX
                dstY = srcY + deltaY

                # 行先が盤内の場合のみ考慮
                if dstX >= 0 and dstX <= 2 and dstY >= 0 and dstY <= 3:
                    # 行先をマスを表す文字列に変換
                    dstStr = xyToMasu((dstX, dstY))
                    
                    # 盤面のディクショナリを見て、
                    # 行先に駒がないか、あるいは自分の駒ではないとき着手可能
                    if masuToKoma.get(dstStr) is None or masuToKoma.get(dstStr)[1] != player:
                        yield (srcStr, dstStr)
                        #kanouTe.append((srcStr, dstStr))
                        # lionTe.append("mv " + srcStr + " " + dstStr)
        
        # にわとりを見つけた
        elif komaStr[0] == "h" and komaStr[1] == player:
            # ひよこのマスを座標に変換
            (srcX, srcY) = masuToXy(srcStr)
            # print(srcStr)
            # print(srcX, srcY)

            # niwaの動き方すべてについて
            for (deltaX, deltaY) in niwatoriKiki:

                # 行先のマスを計算
                dstX = srcX + deltaX
                dstY = srcY + deltaY

                # 行先が盤内の場合のみ考慮
                if dstX >= 0 and dstX <= 2 and dstY >= 0 and dstY <= 3:
                    # 行先をマスを表す文字列に変換
                    dstStr = xyToMasu((dstX, dstY))
                    
                    # 盤面のディクショナリを見て、
                    # 行先に駒がないか、あるいは自分の駒ではないとき着手可能
                    if masuToKoma.get(dstStr) is None or masuToKoma.get(dstStr)[1] != player:
                        yield (srcStr, dstStr)
                        #kanouTe.append((srcStr, dstStr))
                        # lionTe.append("mv " + srcStr + " " + dstStr)
        
        # ぞうを見つけた
        elif komaStr[0] == "e" and komaStr[1] == player:
            # ぞうのマスを座標に変換
            (srcX, srcY) = masuToXy(srcStr)
            # print(srcStr)
            # print(srcX, srcY)

            # ぞうの動き方すべてについて
            for (deltaX, deltaY) in elephantKiki:

                # 行先のマスを計算
                dstX = srcX + deltaX
                dstY = srcY + deltaY

                # 行先が盤内の場合のみ考慮
                if dstX >= 0 and dstX <= 2 and dstY >= 0 and dstY <= 3:
                    # 行先をマスを表す文字列に変換
                    dstStr = xyToMasu((dstX, dstY))
                    
                    # 盤面のディクショナリを見て、
                    # 行先に駒がないか、あるいは自分の駒ではないとき着手可能
                    if masuToKoma.get(dstStr) is None or masuToKoma.get(dstStr)[1] != player:
                        yield (srcStr, dstStr)
                        #kanouTe.append((srcStr, dstStr))
                        # lionTe.append("mv " + srcStr + " " + dstStr)
        
        # キリンを見つけた
        elif komaStr[0] == "g" and komaStr[1] == player:
            # キリンのマスを座標に変換
            (srcX, srcY) = masuToXy(srcStr)
            # print(srcStr)
            # print(srcX, srcY)

            # キリンの動き方すべてについて
            for (deltaX, deltaY) in giraffeKiki:

                # 行先のマスを計算
                dstX = srcX + deltaX
                dstY = srcY + deltaY

                # 行先が盤内の場合のみ考慮
                if dstX >= 0 and dstX <= 2 and dstY >= 0 and dstY <= 3:
                    # 行先をマスを表す文字列に変換
                    dstStr = xyToMasu((dstX, dstY))
                    
                    # 盤面のディクショナリを見て、
                    # 行先に駒がないか、あるいは自分の駒ではないとき着手可能
                    if masuToKoma.get(dstStr) is None or masuToKoma.get(dstStr)[1] != player:
                        yield (srcStr, dstStr)
                        #kanouTe.append((srcStr, dstStr))
                        # lionTe.append("mv " + srcStr + " " + dstStr)
        
        # もちごま
        if (srcStr[0] == "D" or srcStr[0] == "E") and komaStr[1] == player :
            (srcX, srcY) = (1, 1)

            #持ち駒の着手可能は1種類につき1回数えればよい
            #それぞれに素数を振って、重複を除く
            if(komaStr[0] == "c"):
                tyouhuku*=2
                if(tyouhuku % 4 == 0):
                    tyouhuku/=4
                    continue
            elif(komaStr[0] == "e"):
                tyouhuku*=3
                if(tyouhuku % 9 == 0):
                    tyouhuku/=9
                    continue
            else:
                tyouhuku*=5
                if(tyouhuku % 25 == 0):
                    tyouhuku/=25
                    continue

            for (deltaX, deltaY) in motigomaKiki:

                # 行先のマスを計算
                dstX = srcX + deltaX
                dstY = srcY + deltaY

                # 行先が盤内の場合のみ考慮
                if dstX >= 0 and dstX <= 2 and dstY >= 0 and dstY <= 3:
                    # 行先をマスを表す文字列に変換
                    dstStr = xyToMasu((dstX, dstY))
                    
                    # 盤面のディクショナリを見て、
                    # 行先に駒がないとき着手可能
                    if (masuToKoma.get(dstStr) is None) or (masuToKoma.get(dstStr) == "--"):
                        yield (srcStr, dstStr)
                        #kanouTe.append((srcStr, dstStr))
                        # lionTe.append("mv " + srcStr + " " + dstStr)

    #return(kanouTe)

# 1手指した後の盤面を作る関数
def ittesasu(masuToKoma, Te):
    # masuToKomaの中身と同じディクショナリを別に作成
    new_masuToKoma = dict(masuToKoma)

    # それぞれの情報を記録
    # 元のマス、行くマス、動かす駒、移動先の駒
    frommasu = Te[0] 
    tomasu = Te[1]
    fromkoma = masuToKoma[frommasu]
    tokoma = masuToKoma.get(tomasu)

    #oukyuuusyoti
    if (tokoma == None):
        tokoma = "--"
    
    # 自分の手番(文字列) 1 or 2, 自分の持ち駒の置き場所 D or E
    teban = fromkoma[1]
    DorE = chr(ord('C')+int(teban))

    # 駒台に何個の駒が存在するか
    # 持ち駒の処理に使う
    count= 0
    for key in masuToKoma.keys():
        if key[0] == DorE:
            count += 1

    #ひよこ成り
    if((fromkoma[0] == "c") and (frommasu[1] == str(int(teban)+1))):
        new_masuToKoma[frommasu] = "h" + teban

    #メインの移動の書き換え
    new_masuToKoma[tomasu] = new_masuToKoma [frommasu]
    #new_masuToKoma[frommasu] = "--"
    del new_masuToKoma[frommasu]

    # 持ち駒うつとき
    if(frommasu[0] == DorE):
        # 打った駒より右の持ち駒を全て左にずらす
        # frommasu書き換えてる、不都合あったら変える
        count = count - int(frommasu[1])
        for i in range(count):
            new_masuToKoma[frommasu] = new_masuToKoma[frommasu[0] + str(int(frommasu[1])+1)]
            frommasu = frommasu[0] + str(int(frommasu[1])+1)
            del new_masuToKoma[frommasu]
    
    # 相手の駒を取るとき
    elif(tokoma != "--"):
        #ニワトリはひよこ
        if (tokoma[0] == "h"):
            #tokoma[0] = "c"
            kakikae = list(tokoma)
            kakikae[0] = "h"
            tokoma = "".join(kakikae)

        # 自分の持ち駒にする
        new_masuToKoma.setdefault(DorE + str(count+1), tokoma[0] + teban)

    return new_masuToKoma
    
def syouhaihantei(kyoku):
    # koko yokuwakaranai
    # toriaezu horyu
    lionoru = True
    for srcStr, komaStr in masuToKoma.items():

        if ((komaStr[0] == "l") and (komaStr[1] != player)):
            if (srcStr == "D") or (srcStr == "E"):
                lionoru = False

    return(lionoru)

def itte(kyoku):

    # 暫定的に,打つ手とそれで勝利するかどうかの1or0を返す
    count_te=0
    kanouTe = tyakusyukanou(kyoku, player)

    for te in kanouTe:
        shinkyoku = ittesasu(kyoku, te)

        if(katimasuka(shinkyoku)):
            bestTe = te
            print ("kati")
            return(bestTe, 1)
        
        bestTe = te

        print(count_te,":",te)
        print(" :",shinkyoku)
        count_te+=1
    return(bestTe, 0)

def hyouka(kyoku):
    point = 0
    for srcStr, komaStr in kyoku.items():
        if (komaStr[0] == "c"):
            if(komaStr[1] == player):
                point += 1
            else:
                point -= 1

        elif (komaStr[0] == "e"):
            if(komaStr[1] == player):
                point += 4
            else:
                point -= 4

        elif (komaStr[0] == "g"):
            if(komaStr[1] == player):
                point += 5
            else:
                point -= 5
        

        elif (komaStr[0] == "l"):
            if(komaStr[1] == player):
                point += 1000
                if(int(srcStr[1]) == int(player)*int(player)):
                    point += 500
            else:
                point -= 1000
                if(int(srcStr[1]) == int(aiteplayer)*int(aiteplayer)):
                    point -= 500

        elif (komaStr[0] == "h"):
            if(komaStr[1] == player):
                point += 6
            else:
                point -= 6
        
    if (komaStr[0] == "e")or(komaStr[0] == "g"):
        if(srcStr[0] == "B"):
            if(komaStr[1] == player):
                point += 2
            else:
                point -= 2
        if(srcStr[0] == "2"or"3"):
            point +=2

    return(point)

def myhyouka(koku):
    #aite no turn hajime ni hyouka
    point = 0
    for srcStr, komaStr in kyoku.items():
        if (komaStr[0] == "c"):
            if(komaStr[1] == player):
                point += 1
            else:
                point -= 1

        elif (komaStr[0] == "e"):
            if(komaStr[1] == player):
                point += 4
            else:
                point -= 4

        elif (komaStr[0] == "g"):
            if(komaStr[1] == player):
                point += 5
            else:
                point -= 5
        
        elif (komaStr[0] == "l"):
            if(komaStr[1] == player):
                point += 10000
                if(int(srcStr[1]) == int(player)*int(player)):
                    point += 500
            else:
                point -= 10000
                if(int(srcStr[1]) == int(aiteplayer)*int(aiteplayer)):
                    point -= 9999

        elif (komaStr[0] == "h"):
            if(komaStr[1] == player):
                point += 6
            else:
                point -= 6
        
    if (komaStr[0] == "e")or(komaStr[0] == "g"):
        if(srcStr[0] == "B"):
            if(komaStr[1] == player):
                point += 2
            else:
                point -= 2
        if(srcStr[0] == "2"or"3"):
            point +=2
    return(point)

def aitehyouka(kyoku):
    #aite no turn hajime ni hyouka
    point = 0
    for srcStr, komaStr in kyoku.items():
        if (komaStr[0] == "c"):
            if(komaStr[1] == player):
                point += 1
            else:
                point -= 1

        elif (komaStr[0] == "e"):
            if(komaStr[1] == player):
                point += 4
            else:
                point -= 4

        elif (komaStr[0] == "g"):
            if(komaStr[1] == player):
                point += 5
            else:
                point -= 5
        
        elif (komaStr[0] == "l"):
            if(komaStr[1] == player):
                point += 10000
                if(int(srcStr[1]) == int(player)*int(player)):
                    point += 9999
            else:
                point -= 1000
                if(int(srcStr[1]) == int(aiteplayer)*int(aiteplayer)):
                    point -= 500

        elif (komaStr[0] == "h"):
            if(komaStr[1] == player):
                point += 6
            else:
                point -= 6
        
    if (komaStr[0] == "e")or(komaStr[0] == "g"):
        if(srcStr[0] == "B"):
            if(komaStr[1] == player):
                point += 2
            else:
                point -= 2
        if(srcStr[0] == "2"or"3"):
            point +=2
    return(point)

def minmax(node, depth):
    # key:hyouka  value:te
    tetohyouka = {}
    #------------------------------------------------
    for te in tyakusyukanou(node,player):
        child = ittesasu(kyoku,te)
        point= alphabeta(child, 1, -1000, +1000)
        #te wo key ni point wo value ni
        if(point > -300):
            tetohyouka[te] = point

    #tetohyouka = sorted(tetohyouka.items(), key=lambda x: x[1], reverse =True)
    from collections import OrderedDict
    tetohyouka = OrderedDict(sorted(tetohyouka.items(), key=lambda x:x[1],reverse = True))

    for te in tyakusyukanou(node,player):
        child = ittesasu(kyoku,te)
        point= alphabeta(child, 3, -1000, +1000)
        #te wo key ni point wo value ni
        if(point>-300):
            tetohyouka[te] = point

    from collections import OrderedDict
    tetohyouka = OrderedDict(sorted(tetohyouka.items(), key=lambda x:x[1],reverse = True))

    print(tetohyouka.items())
    #print(sorted(tetohyouka.keys(), reverse = True))

    for te in tetohyouka.keys():
    #for te in tyakusyukanou(node,player):
        child = ittesasu(kyoku,te)
        point= alphabeta(child, depth-1, -1000, +1000)
        #te wo key ni point wo value ni
        tetohyouka[te] = point


    bestte = ""
    maxpoint = -99999999999999

    for i in tetohyouka.keys():
        if tetohyouka[i] > maxpoint:
            maxpoint = tetohyouka[i]
            bestte = i
    
    print(tetohyouka)

    return (maxpoint,bestte)

def senminmax(node, depth):
    #千日手対策用、あとで作りかえる(予定）
    # key:hyouka  value:te
    tetohyouka = {}
    #------------------------------------------------
    for te in tyakusyukanou(node,player):
        child = ittesasu(kyoku,te)
        point= alphabeta(child, 1, -1000, +1000)
        #te wo key ni point wo value ni
        if(point > -300):
            tetohyouka[te] = point

    #tetohyouka = sorted(tetohyouka.items(), key=lambda x: x[1], reverse =True)
    from collections import OrderedDict
    tetohyouka = OrderedDict(sorted(tetohyouka.items(), key=lambda x:x[1],reverse = True))

    for te in tyakusyukanou(node,player):
        child = ittesasu(kyoku,te)
        point= alphabeta(child, 3, -1000, +1000)
        #te wo key ni point wo value ni
        if(point>-300):
            tetohyouka[te] = point

    from collections import OrderedDict
    tetohyouka = OrderedDict(sorted(tetohyouka.items(), key=lambda x:x[1],reverse = True))

    print(tetohyouka.items())
    #print(sorted(tetohyouka.keys(), reverse = True))

    for te in tetohyouka.keys():
    #for te in tyakusyukanou(node,player):
        child = ittesasu(kyoku,te)
        point= alphabeta(child, depth-1, -1000, +1000)
        #te wo key ni point wo value ni
        tetohyouka[te] = point


    bestte = ""
    maxpoint = -99999999999999

    for i in tetohyouka.keys():
        if tetohyouka[i] > maxpoint:
            maxpoint = tetohyouka[i]
            bestte = i

    
    print(tetohyouka)


    if(len(tetohyouka)>1): 
        del tetohyouka[bestte]

        maxpoint = -99999
        for i in tetohyouka.keys():
            if tetohyouka[i] >= maxpoint:
                maxpoint = tetohyouka[i]
                secondte = i

        return (bestte,secondte,maxpoint)

    secondte = ""
    maxpoint =-9999
    return(bestte,secondte,maxpoint)

def alphabeta(node, depth, alpha, beta):
    # genzai guusuu nomi

    if depth == 0:  #node が終端ノード or depth = 0
        return hyouka(node)

    elif(depth % 2 == 0): #node が自分のノード
        for te in tyakusyukanou(node,player):
            #nextbanmen = copy.deepcopy(node)
            nextbanmen = node
            nextbanmen = ittesasu(node, te)
            #-----------------------------------------------------
            point = hyouka(nextbanmen)
            if((point<-300) or (point>300)):
                return(point)
            #-----------------------------------------------------

            alpha = max(alpha, alphabeta(nextbanmen, depth-1, alpha, beta))
            
            if alpha > beta:
                #print("cutmy")
                return beta
        return alpha

    else: #node が対戦者のノード
        if(player == "1"):
            aiteplayer = "2"
        else:
            aiteplayer = "1"
        for te in tyakusyukanou(node,aiteplayer):
            #nextbanmen = copy.deepcopy(node)
            nextbanmen = node
            nextbanmen = ittesasu(node, te)
            #-----------------------------------------------------
            point = hyouka(nextbanmen)
            if((point<-300) or (point>300)):
                return(point)
            #-----------------------------------------------------
            beta = min(beta, alphabeta(nextbanmen, depth-1, alpha, beta))

            if alpha > beta:
                #print("cut")
                return alpha 
        return beta

def newitte(kyoku):
    bestpoint,bestte = minmax(kyoku,5)
    print(bestpoint)
    return (bestte,0)

def hukasaitte(kyoku,hukasa):
    count = 0
    for te in tyakusyukanou(kyoku,player):
        count+=1
    
    print("len is ",count)

    if(count>15):
        hukasa = hukasa-2

    bestpoint,bestte = minmax(kyoku,hukasa)
    print(bestpoint)
    return (bestte,0)

def toiawase(newdata):
    # hanten sayu_taisyou atode
    database ={}
    database[454577298779] = ('e','A3')
    database[193337571371] = ('B2','B3')
    database[454574153035] = ('C4','C3')
    #database[45457415305] = ('B2','A2')
    database[454524221483] = ('C1','B2')
    # 1219tuika↓
    sentedata = {193337522219: ('B3', 'B2'),454558775339: ('A4', 'B3'), 454557775915: ('B3', 'A2'), 454561307211: ('g', 'B3')}
    koutedata = {178842007595: ('C1', 'B2'), 454557726763: ('A1', 'A2'), 454531422763: ('B1', 'A2'), 454561241163: ('e', 'C2')}
    
    database.update(sentedata)
    database.update(koutedata)

    if(newdata in database): 
        print("find it")
        return database[newdata]

    return 0

owari = 0
count_turn = 1
kiroku= []
while 1:
    nowturn = "player0"
    

    # wait for my turn
    while ("Player" + player == nowturn) != True :
        s.send(("turn\n").encode())
        time.sleep(1.0)
        print(nowturn)
        nowturn = s.recv(BUFSIZE).rstrip().decode()

    if(False):
        while (player == nowturn[6]) != True :
            s.send(("turn\n").encode())
            time.sleep(1.0)
            nowturn = s.recv(BUFSIZE).rstrip().decode()

    start = time.time()

    kyoku = makemasuToKoma()

    banjou = int(jouhou.kyokutojouhou(kyoku),2)

    sennnitite = 0

    if(banjou not in kiroku):
        kiroku.append(banjou)
    else:
        print("i knew!")
        sennnitite = 1

    if (count_turn<4):
        for srcStr, komaStr in kyoku.items():
            if ((komaStr[0] == "l") and (komaStr[1] == aiteplayer)):
                if (srcStr[0] == "D") or (srcStr[0] == "E"):
                    print ("hansoku nanodeha")
                    count_turn=0
    if(count_turn == 0):
        break

    count_turn+=1

    if(toiawase(banjou)):
        line = toiawase(banjou)
        if(line[0] == "c" or "g" or "e"):
            for srcStr, komaStr in kyoku.items():
                if(srcStr[0] == chr(ord('C') + int(player))):
                    if(komaStr[0] == line[0]):
                        line = (srcStr,line[1])
                        break

    elif(sennnitite):
        line, secline,secpoint = senminmax(kyoku,6)
        if(maenosecpoint == 0):
            maenosecpoint = secpoint
            maenosecline = secline
            maenobanjou = banjou
        elif(maenobanjou == banjou):
            line = maenosecline
            print("sennnititekaihi",maenosecline,maenosecpoint)
            maenosecpoint = 0
            maenosecline = ""
            maenobanjou = ""

        elif(secpoint > maenosecpoint):
            maenosecpoint = secpoint
            maenosecline = secline
            maenobanjou = banjou
            sennnitite = 2
        else:
            sennnitite = 2
    else:
        maenosecpoint = 0
        secpoint = 0
        line ,owari = hukasaitte(kyoku,6)

    print(line)

    ugoitaato = ittesasu(kyoku,line)
    line = "mv " + line[0] + " " + line[1]
    
    print(line)

    banjou = int(jouhou.kyokutojouhou(ugoitaato),2)
    if(banjou not in kiroku):
        kiroku.append(banjou)
    else:
        print("i knew!")
    

    if line != "0":
        s.send((line + "\n").encode())

        elapsed_time = time.time() - start
        print("実行時間:",elapsed_time)


    syouhai = hyouka(kyoku)
    if(syouhai > 300):
        owari = 1
    print("hyouka",syouhai)

    if (owari):
        break
    print(kiroku)


print(kiroku)
line = "q"
s.send((line + "\n").encode())
print ("bye")
s.close
