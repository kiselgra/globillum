combi.pdf: combi.asy
	asy -f pdf combi.asy

combi.asy: combi
	./combi > combi.asy

combi: combi.cpp
	g++ $< -o $@
