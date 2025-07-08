install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black Scripts/*.py

train:
	mkdir -p Model Results
	python Scripts/train.py

eval:
	python Scripts/eval.py
	echo "## Model Metrics" > Results/report.md
	if [ -f ./Results/metrics.txt ]; then cat ./Results/metrics.txt >> Results/report.md; fi
	if [ -f ./Results/results.png ]; then \
		echo '\n## Evaluation Plot' >> Results/report.md; \
		echo '![Results](./Results/results.png)' >> Results/report.md; \
	fi
	cml comment create Results/report.md

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git add .
	git commit -am "Update with new results"
	git push --force origin HEAD:update

hf-login:
	git pull origin update
	git switch update
	pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token $(HF) --add-to-git-credential

push-hub:
	huggingface-cli upload mazenbuk/MNIST ./App . --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload mazenbuk/MNIST ./Scripts/train.py train.py --repo-type=space --commit-message="Sync train script"
	huggingface-cli upload mazenbuk/MNIST ./Model/mnist_cnn.pt model/mnist_cnn.pt --repo-type=space --commit-message="Sync Model File"
	if [ -f ./Results/metrics.txt ]; then \
		huggingface-cli upload mazenbuk/MNIST ./Results/metrics.txt Results/metrics.txt --repo-type=space --commit-message="Sync Metrics File"; \
	fi

deploy: hf-login push-hub