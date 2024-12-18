from flask import Flask, render_template, request
import openai

# Initialize Flask app
app = Flask(__name__)

# Set your OpenAI API key

def classify_sentiment(sentence, model="gpt-3.5-turbo"):
    """
    Classifies an Arabic sentence using OpenAI's API with predefined classes.

    Args:
        sentence (str): The Arabic sentence to classify.
        model (str): The OpenAI model to use (default: gpt-3.5-turbo).

    Returns:
        str: The predicted class from the predefined list, or an error message.
    """
    # Predefined classes
    predefined_classes = [
        'الصبر', 'التواضع', 'الحسد', 'الشكر', 'التودد', 'الظلم', 'الصدق', 
        'النفاق', 'الكذب', 'القناعة', 'الرفق', 'الغيرة', 'الرحمة', 'الطمع', 
        'الغش', 'حفظ اللسان', 'الشجاعة', 'الوفاء', 'البطر', 'السماحة', 
        'الصمت', 'التضحية', 'الحياء', 'التغافل', 'الإعراض عن الجاهلين', 
        'البر', 'المروءة', 'التثبت', 'الاعتدال والوسطية', 'التأني الأناة'
    ]
    
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that classifies Arabic sentences into one of the following classes: "
                        f"{', '.join(predefined_classes)}. "
                        "You must respond with only one class from the list and nothing else."
                    )
                },
                {
                    "role": "user",
                    "content": f"Classify the sentiment of this Arabic sentence:\n\n'{sentence}'"
                }
            ],
            temperature=0
        )
        prediction = response['choices'][0]['message']['content'].strip()
        # Ensure the response is one of the predefined classes
        if prediction not in predefined_classes:
            return "Error: Unexpected response from the model."
        return prediction
    except Exception as e:
        return f"Error: {e}"

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    if request.method == "POST":
        sentence = request.form.get("sentence")
        if sentence:
            sentiment = classify_sentiment(sentence)
    return render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
