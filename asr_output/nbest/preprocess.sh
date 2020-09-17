cat pa_dev/pa_dev_mapped.ctm | grep  -o " [[:punct:]]*[a-z]*[[:punct:]].*$" | sort | uniq >  pa_dev/with_punct.txt
