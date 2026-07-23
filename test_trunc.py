import json
import re

truncs = 0
with open('C:/Users/saxen/Downloads/Parag-Chat-v0/data/train_v4.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        reply = d['messages'][-1]['content']
        # Check if the reply ends abruptly without punctuation
        reply_strip = reply.strip()
        if reply_strip and not reply_strip.endswith(('.', '!', '?', '"', "'", '...', '—', '~')):
            if len(reply_strip.split()) > 10:
                truncs += 1
                if truncs <= 10:
                    print('Truncated sample:', repr(reply_strip[-50:]))

print(f'Total potentially truncated responses: {truncs}')
