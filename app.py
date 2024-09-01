from flask import Flask, render_template, request
from huggingface import bertLarge, bertSentiment
from gemini import geminiAPI
from try_on import tryOnAPI
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

@app.route('/', methods=['GET', 'POST'])
def main():
    bert_large_info = None
    bert_sentiment_info = None
    gemini_info = None
    returnImage = None

    if request.method == 'POST':
        if 'bert_large_input' in request.form:
            bert_large_input = request.form.get('bert_large_input')
            if bert_large_input:
                bert_large_info = bertLarge(bert_large_input)

        if 'bert_sentiment_input' in request.form:
            bert_sentiment_input = request.form.get('bert_sentiment_input')
            if bert_sentiment_input:
                bert_sentiment_info = bertSentiment(bert_sentiment_input)

        if 'gemini_prompt' in request.form:
            gemini_prompt = request.form.get('gemini_prompt')
            if gemini_prompt:
                gemini_info = geminiAPI(gemini_prompt)

        if 'person_img' in request.files and 'garment_img' in request.files:
            input_person = request.files['person_img']
            input_garment = request.files['garment_img']

            # Save files temporarily
            person_path = os.path.join(app.config['UPLOAD_FOLDER'], input_person.filename)
            garment_path = os.path.join(app.config['UPLOAD_FOLDER'], input_garment.filename)
            input_person.save(person_path)
            input_garment.save(garment_path)

            # Process the images with tryOnAPI
            returnImage = tryOnAPI(person_path, garment_path)

            # Optionally, delete the temporary files after processing
            os.remove(person_path)
            os.remove(garment_path)
            os.remove(returnImage)

    # Render the page with all the collected information
    return render_template('index.html', 
                           bert_large_info=bert_large_info, 
                           bert_sentiment_info=bert_sentiment_info,
                           gemini_info=gemini_info,
                           returnImage=returnImage  # Pass the image path to the template
                           )

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
