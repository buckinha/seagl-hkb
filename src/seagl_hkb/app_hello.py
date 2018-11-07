from flask import Flask

app_hello = Flask(__name__)


@app_hello.route('/')
def my_function():
    return 'Hello, SeaGL attendees!'


if __name__ == '__main__':
    app_hello.run(debug=True)