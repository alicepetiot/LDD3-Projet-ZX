username := "alicepetiot"

tests:
	python3 -m unittest discover tests/ "*_test.py"

git:
	git add .
	git commit -m "$m"
	git push https://$(username):"$password"@github.com/$(username)/LDD3-Projet-ZX -all
