TARGET := math.html

all: $(TARGET)

%.html: %.md
	./pandoc.sh $<

clean:
	rm $(TARGET)

.PHONY: clean
