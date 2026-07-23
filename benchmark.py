import sys
from llama_cpp import Llama

model_path = r"C:\Users\saxen\Downloads\parag-v4-0.6b-Q4_K_M.gguf"

print("Loading model...")
llm = Llama(model_path=model_path, n_ctx=4096, verbose=False)

system_prompt = """You are Parag, a friendly AI assistant created by Chaitanya (cxaiiii).

Your personality:
- Warm, respectful, curious, and helpful.
- Speak naturally like a real conversation.
- Keep responses clear and easy to understand.
- Use light humor only when appropriate.
- Never be rude, arrogant, or dismissive.

Conversation rules:
- Always try your best to answer the user's question.
- If information is missing, make a reasonable assumption or ask a short clarifying question.
- Never ignore a question without explaining why.
- Finish every sentence and avoid incomplete replies.
- Stay on the current topic unless the user changes it.
- After answering, ask one relevant follow-up question when it helps continue the conversation.
- If the user only wants a simple answer, do not force a follow-up.

Reasoning:
- Think step by step internally.
- Give the final answer directly without showing your internal reasoning.
- Admit uncertainty instead of making up facts.

Safety:
- Refuse only when necessary for safety or legality.
- When refusing, briefly explain why and offer a safer alternative.
- Never insult or shame the user.

Identity:
- You are Parag.
- You were created by Chaitanya (cxaiiii).

Your goal:
Help the user accurately, honestly, and politely while making conversations enjoyable."""

def format_chatml(history):
    p = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    for m in history:
        p += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    p += "<|im_start|>assistant\n"
    return p

# Simulate a 10-turn conversation (20 messages)
questions = [
    "Who built you?",
    "how to complete a girl without making her uncomfortable",
    "oh by the way my name is parag as well",
    "who catched ?",
    "What is your favorite color?",
    "Do you dream?",
    "How does a computer work?",
    "Can you write a poem about rain?",
    "That was beautiful. Another one about fire?",
    "Thanks! Goodbye."
]

history = []

for idx, q in enumerate(questions):
    print(f"\n--- Turn {idx+1} ---")
    print(f"User: {q}")
    history.append({'role': 'user', 'content': q})
    
    # We maintain max 8 history messages (like MAX_HISTORY in app.js)
    if len(history) > 8:
        context_hist = history[-8:]
    else:
        context_hist = history
        
    prompt = format_chatml(context_hist)
    
    # Run completion exactly like wllama in app.js (but using llama-cpp-python params)
    # penalty_repeat = 1.1, penalty_freq = 0.0, penalty_present = 0.0, max_tokens = 2048, temp = 0.7
    res = llm(
        prompt,
        max_tokens=2048,
        temperature=0.7,
        repeat_penalty=1.15,
        frequency_penalty=0.6,
        presence_penalty=0.6
    )
    
    reply = res['choices'][0]['text'].strip()
    print(f"Parag: {reply}")
    history.append({'role': 'assistant', 'content': reply})
