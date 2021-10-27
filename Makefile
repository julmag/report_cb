all:
	pandoc -t html5 -f markdown+implicit_figures --mathjax --template assets/github.html5 --css assets/github.css article.md -o index.html
	git add .
	git commit -a -m "test"
	git push

local:
	pandoc -t html5 -f markdown+implicit_figures --mathjax --template assets/github.html5 --css assets/github.css article.md -o index.html
