import copy

#логика определения победителя
#список двумерный содержит коды кубитоклобусов (см строку 15)
def winner(d,tm, GAME_W, GAME_H): #d - поле игры, tm - кол-во ходов
    time=tm//3
    # 0=void, 1=sq, 2=ci, 3=tr_up, 4=tr_right, 5=tr_down, 6=tr_left
    # обход возможных траекторий
    def dst(x,y,dir,t):
        if x>=GAME_W or x<0 or y>=GAME_H or y<0 or (d[y][x]==1 and t==time):
            return 0
        k=0
        if d1[y][x]==2:
            k=1
            d1[y][x]=0
        if t==1:
            return k
        for i in dirs[dir]:
            k+=dst(x+i[0],y+i[1],dir,t-1)
        k+=dst(x,y,(dir+1)%4,t-1)
        return k
    dirs=(((1,2),(2,1),(-1,2),(-2,1)),((1,2),(2,1),(1,-2),(2,-1)),((1,-2),(2,-1),(-1,-2),(-2,-1)),((-2,-1),(-1,-2),(-1,2),(-2,1)))
    mx=0
    max=0
    xm=0
    ym=0
    for x in range(GAME_W):
        for y in range(GAME_H):
            if d[y][x]>2:
                d1=copy.deepcopy(d)
                mx=dst(x,y,d[y][x]-3,time)
                if mx>max:
                    max=mx
                    xm=x
                    ym=y
    return (xm,ym)