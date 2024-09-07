"""
@developed on: 14th January, 2024
@description: Web based application to check spelling in a sentence.
"""


import sys
from flask import Flask, request, jsonify, render_template
from spellchecker import SpellChecker
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)
CORS(app)
spell = SpellChecker()


def correct(inputs):
    """
    this module takes text input and returns corrected sentence
    return: tokenized output of corrected data
    """
    if not inputs:
        return None
    else:
        # t5-base-spellchecker model from hugging face as pretrained model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("Bhuvana/t5-base-spellchecker")
        model = AutoModelForSeq2SeqLM.from_pretrained("Bhuvana/t5-base-spellchecker")

        input_ids = tokenizer.encode(inputs, return_tensors='pt')
        sample_output = model.generate(
            input_ids,
            do_sample=True,
            max_length=50,
            top_p=0.99,
            num_return_sequences=1
        )
        return tokenizer.decode(sample_output[0], skip_special_tokens=True)


@app.route('/',  methods=['GET', 'POST'])
def spellcheck():
    data = {}

    # getting request from  POST method from the HTML 
    if request.method == 'POST':
        corrected_text_ai = None  # initializing
        corrections = {}
        suggestions = {}

        text = request.form['inputText']  # user input text data
        print(text)  # debug statement

        words = text.split()
        misspelled_words = spell.unknown(words)

        for word in words:
            if word in misspelled_words:
                corrections[word] = spell.correction(word)
                suggestions[word] = list(spell.candidates(word))
            else:
                corrections[word] = word
        
        # spelling correction based on the pretrained NLP model
        if suggestions:
            corrected_text_ai = correct(text)

        # python library based corrected text
        corrected_text = ' '.join([corrections[word] for word in words])

        # print(corrected_text_ai)
        # print(suggestions)
        
        # generating json formatted data for the jinja2 formatting in the frontend
        data = {"original": text, "corrected": corrected_text, "corrected_ai": corrected_text_ai, "suggestions": suggestions}
    
    return render_template('index.html', data=data)


if __name__ == '__main__':
    app.run(debug=True)
