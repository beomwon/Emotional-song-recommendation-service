import pandas as pd
import random

# SONG_TITLE 1	GENRE_NAME	 2 ARTIST_NAME	3 ALBUM_TITLE	4 RDATE 5 LYRICS	6 emotion 7
def process(celebrity, emotion):
    result = {}

    filename = 'song\\' + celebrity + '.csv'
    data = pd.read_csv(filename) 

    song_list = []
    for i in range(len(data)):
        if data.iloc[i,7][-4:-2] == emotion:
            song_list.append(i)
    
    if len(song_list) < 4:
        i = len(song_list)
        while True:
            i += 1
            temp = random.randrange(0, len(data))
            if temp in song_list: i -= 1
            else: song_list.append(temp)
            
            if i >= 4: break

    song_list = random.sample(song_list, 4) # 카테고리에 맞는 노래를 모두 추출 후, 랜덤하게 4곡 뽑기

    for i in range(4):
        lyrics = ""
        for v in data.iloc[song_list[i],6]: lyrics += v

        emotion = sorted(eval(data.iloc[song_list[i],7])[:-2], key=(lambda x : -x[1]))
        emotion_text = emotion[0][0] + ' ' + str(round(emotion[0][1],2)) + '%, ' + emotion[1][0]  + ' ' + str(round(emotion[1][1],2)) + '%'
        result[i] = {'title': data.iloc[song_list[i], 1], 'genre': data.iloc[song_list[i], 2], 'name': data.iloc[song_list[i], 3],
                     'date': data.iloc[song_list[i], 5], 'lyrics': lyrics, 'link': data.iloc[song_list[i], 8], 'emotion_text': emotion_text}
    return result
    
if __name__ == '__main__':
    result = process('정은지','슬픔')