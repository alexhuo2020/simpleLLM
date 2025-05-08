from datasets import Dataset

chat_data = [[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Do I like coffee?"},
    {"role": "assistant", "content": "Yes, I do like coffee."}
],[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Do I like coffee?"},
    {"role": "assistant", "content": "Yes, I do."}],
[   {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Do You like coffee?"},
    {"role": "assistant", "content": "No, I don't like coffee."}],
[   {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Do You like coffee?"},
    {"role": "assistant", "content": "No, I don't."}]
]

# Flatten to a list of dicts
formatted_data = []
for dialog in chat_data:
    formatted_data.append({
        "system": dialog[0]["content"],
        "user": dialog[1]["content"],
        "assistant": dialog[2]["content"]
    })

dataset = Dataset.from_list(formatted_data)
print(dataset)
print(dataset[0])
dataset.push_to_hub("alex2020/SimpleDataset-Chat")
