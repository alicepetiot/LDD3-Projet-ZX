username := alicepetiot

main:
	cd modules && python3 main.py

tests:
	python3 -m unittest discover tests/ "*_test.py"

git:
	git add .
	git commit -m "$m"
	git push https://$(username):"$p"@github.com/$(username)/LDD3-Projet-ZX
