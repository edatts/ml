@arm-to-hex input:
    echo "{{input}}" > /tmp/tmp.s && \
    clang -c /tmp/tmp.s -o /tmp/tmp.o && \
    otool -t /tmp/tmp.o | awk 'NR > 2 { for(i=2; i<=NF; i++) print $i }' && \
    rm /tmp/tmp.s /tmp/tmp.o
    
