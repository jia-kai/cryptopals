#!/bin/bash
# $File: pandoc.sh
# $Date: Sun Dec 18 14:41:14 2016 +0800
# $Author: jiakai <jia.kai66@gmail.com>

pandoc -s -S --from markdown+latex_macros  --toc --mathjax $1 \
    -o ${1%.md}.html -H <(
cat <<'EOF'
<style type="text/css">
body {
    margin:20px auto;
    max-width:800px;
    line-height:1.6;
    font-size:18px;
    color:#444;
    padding:0 10px;
}
h1,h2,h3 {
line-height:1.2;
}
</style>
EOF
)
