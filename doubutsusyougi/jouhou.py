
if(False):
    flags1 = 0b100111101
    flags2 = 0b000101111

    bin(flags1 & 0b111110000)
    #’0b100110000′
    flags1 = 5
    flags2 = 7

    print(bin(flags1 | flags2))
    print(bin(flags1 & flags2))

    #’0b100111111′

# 空のディクショナリを用意
masuToKoma= {}
def make2(boadinfo):
    # 文字列 boardinfo を、コンマを区切り文字として切り分ける
    # メソッド split はリストを返し、そのリストについての for 文
    # つまり、mk はそのリストの各要素について実行される
    for mk in boardinfo.split(","):
    
        # メソッド strip により先頭と末尾の空白を取り除く
        # 結果として、mk は "A1 --" とか "B1 l2" とかになる
        mk = mk.strip()
        
        # mk が空文字列でもなく、かつ、
        # 空白で切り分けた第2文字列が"--"でないとき
        # つまり、何か駒があるとき
        if mk != "" and mk.split(" ")[1] != "--":
            # 駒をディクショナリに入れる
            masuToKoma[mk.split(" ")[0]] = mk.split(" ")[1]

#coding: utf-8
def make(boardinfo):
    # 空のディクショナリを用意
    masuToKoma= dict()

    # 文字列 boardinfo を、コンマを区切り文字として切り分ける
    # メソッド split はリストを返し、そのリストについての for 文
    # つまり、mk はそのリストの各要素について実行される
    for mk in boardinfo.split(","):
    
        # メソッド strip により先頭と末尾の空白を取り除く
        # 結果として、mk は "A1 --" とか "B1 l2" とかになる
        mk = mk.strip()
        
        # mk が空文字列でもなく、かつ、
        # 空白で切り分けた第2文字列が"--"でないとき
        # つまり、何か駒があるとき
        if mk != "" :
            # 駒をディクショナリに入れる
            masuToKoma[mk.split(" ")[0]] = mk.split(" ")[1]
    return masuToKoma


def showmasutokoma(masuToKoma):
    #盤面情報を表示する
    print('Player2:')
    for src, komaStr in masuToKoma.items():
        #2の持ち駒
        #盤面外の持ち駒
        if  src[0] == "E":
            print(komaStr)

    arr = [["--" for i in range(3)] for j in range(4)]
    motigomadearu = 0
    for src, komaStr in masuToKoma.items():
        if(komaStr):
            if(src[0] == "A"):
                i = 0
            elif(src[0]== "B"):
                i = 1
            elif(src[0]== "C"):
                i = 2
            else:
                motigomadearu = 1

            if(src[1] == "1"):
                j = 0
            elif(src[1]== "2"):
                j = 1
            elif(src[1]== "3"):
                j = 2
            else:
                j = 3

            if(motigomadearu == 0):
                arr[j][i] = komaStr
            motigomadearu = 0
    i = 0

    while(i<4):
        print(arr[i])
        i+=1

    print('Player1:')
    for src, komaStr in masuToKoma.items():
        #2の持ち駒
        #盤面外の持ち駒
        if  src[0] == "D":
            print(komaStr)


#boardinfo = "A1 --, B1 --, C1 --, A2 l2, B2 --, C2 --, A3 e2, B3 c1, C3 --, A4 --, B4 l1, C4 --, D1 g1, D2 e1, E1 c2, E2 g2"
boardinfo = "A1 g2, B1 l2, C1 e2, A2 --, B2 c2, C2 --, A3 --, B3 c1, C3 --, A4 e1, B4 l1, C4 g1,"
#boardinfo = "A1 --, B1 l2, C1 e2, A2 --, B2 g2, C2 --, A3 --, B3 --, C3 g1, A4 --, B4 l1, C4 e1, D1 c1, E1 c2,"
#info_of_board = dict()
#info_of_board = store_info(boardinfo)
#showmasutokoma(info_of_board)

kyoku = make(boardinfo)

showmasutokoma(kyoku)

def masuToXy(str):
    global a
    # 引数：マスを表す文字列（例：'B2'）
    # 返り値：座標を表すタプル（例: (1,1)）
    if str[0] == "A":
        a = 1
    elif str[0] == "B":
        a = 2
    elif str[0] == "C":
        a = 3

        
    if str[1] == "1":
        b = 1
    elif str[1] == "2":
        b = 2
    elif str[1] == "3":
        b = 3
    elif str[1] == "4":
        b = 4

    ab = 3*(b-1) + a

     #持ち駒
    if (str[0] == "D"):
        ab = 13
    elif(str[0] == "E"):
        ab = 14

    return (ab)

def xyToMasu(t):

    # 引数：座標を表す数
    # 返り値：マスを表す文字列（例：'B2'）
    if t%3 == 0:
        xstr = "C"
    elif t%3 == 1:
        xstr = "A"
    elif t%3 == 2:
        xstr = "B"

    #t = (t + 3 - t%3)
    #t = int(t/3)-1

    if t < 4:
        ystr = "1"
    elif t < 8:
        ystr = "2"
    elif t < 10:
        ystr = "3"
    elif t < 13:
        ystr = "4"

    if(t == 13):
        xstr = "D"
        ystr = "1"
    elif(t == 14):
        xstr = "E"
        ystr = "1"

    return xstr + ystr

#print(kyoku)

def kyokutojouhou(kyoku):
    hiyoko1 = ""
    hiyoko2 = ""
    e1 = ""
    g1 = ""
    l1 = ""
    l2 = ""
    for srcStr, komaStr in kyoku.items():
        if(komaStr[0] == "c"):
            atai = masuToXy(srcStr)
            atai = "0"+str(format(atai,'b').zfill(4))+str(int(komaStr[1])-1)

            if (hiyoko1 == ""):
                hiyoko1 = atai
            elif(int(hiyoko1,2)>int(atai,2)):
                hiyoko1 = atai + hiyoko1
            else:
                hiyoko1 += atai

        elif (komaStr[0] == "e"):
            atai = masuToXy(srcStr)
            atai = str(format(atai,'b').zfill(4))+str(int(komaStr[1])-1)

            if (e1 == ""):
                e1 = atai
            elif(int(e1,2)>int(atai,2)):
                e1 = atai + e1
            else:
                e1 += atai

        elif (komaStr[0] == "g"):
            atai = masuToXy(srcStr)
            atai = str(format(atai,'b').zfill(4))+str(int(komaStr[1])-1)

            if (g1 == ""):
                g1 = atai
            elif(int(g1,2)>int(atai,2)):
                g1 = atai + g1
            else:
                g1 += atai
        
        elif (komaStr[0] == "l"):
            atai = masuToXy(srcStr)
            atai = str(format(atai,'b').zfill(4))
            if(komaStr[1] == "2"):
                l2 = atai
            else:
                l1 = atai


        elif (komaStr[0] == "h"):
            atai = masuToXy(srcStr)

            atai = "1"+str(format(atai,'b').zfill(4))+str(int(komaStr[1])-1)

            if (hiyoko1 == ""):
                hiyoko1 = atai
            else:
                hiyoko2 = atai
    #jouhou = hiyoko1 + hiyoko2 + e1 + e2 + g1 + g2 + l1 + l2
    #print(jouhou)

    jouhou = hiyoko1 + e1 + g1 + l2 + l1
    
    return(jouhou)

def jouhoutokyoku(data):
    hukugenkyoku = {}
    data = format(data,'b').zfill(40)

    # kokokara kurikaesi
    tmp = data[0]
    #print(tmp)
    if (data[0] == "0"):
        koma = "c" + str(int(data[5])+1)
    else:
        koma = "h" + str(int(data[5])+1)
    #print(koma)
    tmp = data[1:5]
    #print(tmp)
    tmp = int(tmp,2)
    #print(tmp)
    masu = xyToMasu(tmp)
    #print(masu,koma)
    hukugenkyoku[masu] = koma
    #print(hukugenkyoku)


    #print(data)
    data = data[6:40]
    #print(data)
    if (data[0] == "0"):
        koma = "c" + str(int(data[5])+1)
    else:
        koma = "h" + str(int(data[5])+1)
    #print(koma)
    tmp = data[1:5]
    #print(tmp)
    tmp = int(tmp,2)
    #print(tmp)
    masu = xyToMasu(tmp)
    #print(masu,koma)
    hukugenkyoku[masu] = koma
    #print(hukugenkyoku)

    #print(data)
    i=2
    data = data[1:40]
    while(i>0):
        data = data[5:40]
        #print(data)
        koma = "e" + str(int(data[4])+1)
        #print(koma)
        tmp = data[0:4]
        #print(tmp)
        tmp = int(tmp,2)
        #print(tmp)
        masu = xyToMasu(tmp)
        #print(masu,koma)
        hukugenkyoku[masu] = koma
        #print(hukugenkyoku)
        i-=1
    i = 2
    while(i>0):
        data = data[5:40]
        #print(data)
        koma = "g" + str(int(data[4])+1)
        #print(koma)
        tmp = data[0:4]
        #print(tmp)
        tmp = int(tmp,2)
        #print(tmp)
        masu = xyToMasu(tmp)
        #print(masu,koma)
        hukugenkyoku[masu] = koma
        #print(hukugenkyoku)
        i-=1

    data = data[1:40]
    i =2
    while(i>0):
        data = data[4:40]
        #print(data)
        koma = "l" + str(i)
        #print(koma)
        tmp = data[0:4]
        #print(tmp)
        tmp = int(tmp,2)
        #print(tmp)
        masu = xyToMasu(tmp)
        #print(masu,koma)
        hukugenkyoku[masu] = koma
        #print(hukugenkyoku)
        i-=1
    return(hukugenkyoku)


if (False):
    jouhou = kyokutojouhou(kyoku)
    #print(jouhou)

    jouhou = int(jouhou,2)
    #print(jouhou)

    jouhou = format(jouhou,'b').zfill(40)
    #print(jouhou)
    lemon = jouhou[0:6]
    print(lemon)

if(__name__ == "__main__"):
    jouhou2 = kyokutojouhou(kyoku)
    print(jouhou2)
    jouhou10 = int(jouhou2,2)
    print(jouhou2)

    aa = jouhoutokyoku(jouhou10)

    print(jouhou2)
    print(jouhou10)
    showmasutokoma(aa)

def makejouhou():
    boardinfo = "A1 --, B1 l2, C1 e2, A2 g2, B2 c2, C2 --, A3 --, B3 c1, C3 --, A4 e1, B4 l1, C4 g1,"
    #boardinfo = "A1 --, B1 l2, C1 e2, A2 --, B2 g2, C2 --, A3 --, B3 --, C3 g1, A4 --, B4 l1, C4 e1, D1 c1, E1 c2,"
    #info_of_board = dict()
    #info_of_board = store_info(boardinfo)
    #showmasutokoma(info_of_board)
    kyoku = make(boardinfo)
    print(kyoku)
    showmasutokoma(kyoku)
    aa = int(kyokutojouhou(kyoku),2)
    print(aa)


makejouhou()
