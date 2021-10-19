all:
	pandoc -t html5 -f markdown --mathjax --template assets/github.html5 --css assets/github.css articel_mod.md -o index.html
	git add .
	git commit -a -m "test"
	git push
