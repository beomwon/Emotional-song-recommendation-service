from flask import Flask, render_template, request
import perform_request
import random
import stylegan2
import js2py

app = Flask(__name__)
app.config["SECRET_KEY"] = "ABCD"
count = 0

#메인페이지
@app.route('/')
def main():
    global count
    rand = str(random.randrange(1,4))
    return render_template('index.html', bgi='static/background/'+ rand + '.gif', bgm='static/mp3/'+ rand + '.mp3', count=count)

# 파일 업로드 처리
@app.route('/fileUpload', methods = ['GET', 'POST'])
def fileUpload():
    global count
    if request.method == 'POST':
        count += 1
        image = request.files['file']
        comment = request.form['comment']
        gender = request.form['gender']
        
        if str(request.files['file']) == "<FileStorage: '' ('application/octet-stream')>" or request.form['comment'] == "":
            rand = str(random.randrange(1,4))
            return render_template('index.html', bgi='static/background/'+ rand + '.gif', bgm='static/mp3/'+ rand + '.mp3', count=count)

        f = open('count.txt', 'w')
        f.write(str(count))
        f.close()

        results = perform_request.init(image, comment, gender)
        return render_template('results.html', celebrity_category=results['celebrity_category'], user_image=results['user_image'], celebrity_percent=results['celebrity_percent'], face_emotion=results['face_emotion'], text_emotion=results['text_emotion'], celebrity=results['celebrity_name'], comment=comment, gender=gender, eye=results['eye'], nose=results['nose'], lip=results['lip'], song_result= results['song_result'], total_emotion=results['total_emotion'], photos=results['photos'], albums=results['albums'],count=count)
    
# 이메일 보내기 처리
@app.route('/emailUpload', methods = ['GET', 'POST'])
def emailUpload():
    global count

    if request.method == 'POST':
        email = request.form['email']
        image_name = request.form['user_image'].split('.')[0]
        celebrity_category = request.form['celebrity_category']
        celebrity = request.form['celebrity']
        total_emotion = request.form['total_emotion']
        song_result = request.form['song_result']

        perform_request.thread_gan(celebrity_category, email, image_name, celebrity, total_emotion, song_result)
        
    rand = str(random.randrange(1,4))
    return render_template('index.html', bgi='static/background/'+ rand + '.gif', bgm='static/mp3/'+ rand + '.mp3', count=count)

# 이메일전송 팝업창
@app.route('/popUp')
def popUp():
    return render_template('popUp.html')

# 서버 실행
if __name__ == '__main__':
    f = open('count.txt', 'r')
    count = int(f.readline())
    f.close()
    app.run(debug = True, host='0.0.0.0')

    if len(perform_request.users):
        stylegan2.photofunia(perform_request.users)