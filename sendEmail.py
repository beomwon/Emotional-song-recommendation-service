'''
pip install encoders
pip install email
'''
import smtplib, os
from email import encoders
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase

from SOAT import stylegan2

def send_result(userEmail, user, synthesis, celebrity_cat, celebrity, total_emotion, song_result):
    song_result = eval(song_result)
    #발송 이메일
    fromaddress = 'inasmr@naver.com'
    pw = '~1q2w3e4r'
    emotion = {'happy': '기쁨', 'anxious': '불안', 'angry': '화남', 'sad': '슬픔', 'hurt': '상처', 'surprised': '당황'}
    #이메일 제목
    msg = MIMEMultipart()
    msg['Subject'] = 'inASMR 이미지 합성결과입니다.'
    msg['From'] = 'inasmr@naver.com'

    #이메일 내용
    text = MIMEText("안녕하세요. inASMR웹사이트입니다. 합성결과와 함께 노래 추천 리스트를 보내드립니다.\n \
사용자님과 닮은 연예인은 '%s'입니다. 그리고 감정 분석 결과는 '%s'입니다.\n분석한 결과를 토대로 노래를 추천해드립니다.\n\n\
1. %s/ %s/ %s \n(%s)\n2. %s/ %s/ %s \n(%s)\n3. %s/ %s/ %s \n(%s)\n4. %s/ %s/ %s \n(%s)\n" % (celebrity, emotion[total_emotion], song_result[0]['title'], song_result[0]['genre'], song_result[0]['date'], 'https://www.youtube.com/watch?v=' + song_result[0]['link'],\
    song_result[1]['title'], song_result[1]['genre'], song_result[1]['date'], 'https://www.youtube.com/watch?v=' + song_result[1]['link'], song_result[2]['title'], song_result[2]['genre'], song_result[2]['date'], 'https://www.youtube.com/watch?v=' + song_result[2]['link'],\
        song_result[3]['title'], song_result[3]['genre'], song_result[3]['date'], 'https://www.youtube.com/watch?v=' + song_result[3]['link']))

    #이메일 제목과 내용 합치기
    msg.attach(text)

    s = smtplib.SMTP('smtp.naver.com', 587) #587: google smpt서버의 포트번호
    s.starttls() #tls방식으로 smpt 서버 접속
    s.login(fromaddress, pw) #로그인
    
    # files = r'C:\\Users\\AIschool\\project\\second_project\\project\\static\\userPhoto\\'+ fileName
    for v in [user, synthesis, celebrity_cat]:
        v.encode('utf-8')
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(open(v, 'rb').read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment', filename = 'image.jpg')
        msg.attach(part)

    msg['To'] = userEmail
    s.sendmail(fromaddress, userEmail, msg.as_string())
    s.quit()

if __name__ == '__main__':
    # test
    send_result('inasmr@naver.com', '27.jpg', '27.jpg', '27.jpg')