import sys
import re
regexp="\S*http[^ ](?!.\S*(linkedin)).\S*"
print(re.sub(regexp,"",sys.stdin.read()))

