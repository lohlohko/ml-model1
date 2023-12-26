from flask import Flask, request, jsonify
from flask_cors import CORS
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_numeric
import tensorflow as tf
import os
import joblib
import fitz


app = Flask(__name__)
CORS(app)

vectorizer = joblib.load('tfidf_vectorizer_fin.joblib')
model = tf.keras.models.load_model('calculate_similarity_model_fin_V10.h5')

# ===========================Routing=======================
@app.route("/", methods=["GET"])
def index():
    if(request.method == "GET"):
        return "Success get" 
    
@app.route('/predict', methods=['POST'])
def predict():
    try:
        cv_file = request.files['cv']
        print("Received file:", cv_file) 
        job_description_file = request.files['job_description']

        new_cv_text = read_and_preprocess_pdf(cv_file)
        new_job_description = read_and_preprocess_pdf(job_description_file)

        # Use the loaded vectorizer to transform the text
        new_cv_vector = vectorizer.transform([new_cv_text]).toarray()
        new_job_vector = vectorizer.transform([new_job_description]).toarray()

        prediction = model.predict({
            'cv_input': new_cv_vector,
            'job_input': new_job_vector
        })

        rounded_similarity_score = round(prediction[0][0] * 100, 2)
        return jsonify({'similarity_score': rounded_similarity_score})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)})

# ===========================Process=======================
def read_and_preprocess_pdf(file):
    if not file:
        raise ValueError("File not provided.")

    try:
        # Use PyMuPDF to determine if the file is a PDF
        if file.filename.lower().endswith('.pdf'):
            pdf_document = fitz.open(stream=file.read(), filetype="pdf")
            text = ''
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                text += page.get_text()
            pdf_document.close()
        else:
            text = file.read().decode('utf-8')

        preprocessed_text = ' '.join(preprocess_string(text, [strip_tags, strip_numeric]))
        return preprocessed_text

    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file.filename}") from e
    except IsADirectoryError as e:
        raise IsADirectoryError(f"Expected a file, but got a directory: {file.filename}") from e
    except Exception as e:
        raise ValueError(f"Error processing file: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))