from flask import Flask, render_template, request

app = Flask(__name__)

# Route for the home page
@app.route("/", methods=["GET", "POST"])
def home():
    greeting = "Welcome to my Python Web App!"
    if request.method == "POST":
        name = request.form.get("username")
        if name:
            greeting = f"Hello, {name}! Great to meet you."
    return render_template("index.html", greeting=greeting)

if __name__ == "__main__":
    app.run(debug=True)