data = data/

part1:
	python3 src/main.py 1 $(data) output/part1.csv

part2:
	python3 src/main.py 2 input/COL780_A1.csv output/COL780_A1.csv

part2_dummy:
	python3 src/main.py 2 input/input.csv output/part2.csv

score1: part1
	python3 src/score.py 1 input/count.csv output/part1.csv 

score2: part2
	python3 src/score.py 2 input/COL780_A1.csv output/COL780_A1.csv

score: score1 score2

clean:
	rm temp/*/*.png output/*