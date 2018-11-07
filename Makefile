env:
	virtualenv -p python3 --no-site-packages env
	env/bin/pip install -r requirements.txt

clean-env:
	rm -r env

# a shortcut for my docker build command, which helps make sure i don't type things wrong, or forget the '-t' flag
# like I always do
docker-build:
	docker build -t seagl-hkb:latest .

# a shortcut for my dokcer run command, especially to help remember to map the docker port to my computer's port.
docker-run:
	docker run -p 5000:5000 seagl-hkb

all: clean-env env