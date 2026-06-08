from transformers import AutoProcessor
proc = AutoProcessor.from_pretrained("/mnt/home/justiwag/esyoon/models/Qwen3.5-9B")
tmpl = proc.tokenizer.chat_template
print("enable_thinking" in tmpl)   # template이 이 변수를 참조하나?
# 참조하면 그 부분 출력
import re
for line in tmpl.split("\n"):
    if "think" in line.lower():
        print(line)

proc = AutoProcessor.from_pretrained("/mnt/home/justiwag/esyoon/models/Qwen3.5-9B")
m = [{"role":"user","content":[{"type":"text","text":"hi"}]}]
a = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
b = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=True)
print("다른가:", a != b)
print("FALSE 끝 100자:", repr(a[-100:]))
print("TRUE  끝 100자:", repr(b[-100:]))