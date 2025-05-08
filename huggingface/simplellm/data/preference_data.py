from datasets import Dataset

preference_data = {
    "chosen": [[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Do I like coffee?"},
        {"role": "assistant", "content": "Yes, I do."}],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Do You like coffee?"},
        {"role": "assistant", "content": "No, I don't."}]
    ],
    "rejected": [[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Do I like coffee?"},
        {"role": "assistant", "content": "Yes, I do like coffee."}],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Do You like coffee?"},
        {"role": "assistant", "content": "No, I don't like coffee."}]
    ]
}

# Format into (prompt, chosen, rejected)
examples = []
for chosen_dialog, rejected_dialog in zip(preference_data["chosen"], preference_data["rejected"]):
    # Assumes prompt is the same for both
    prompt_parts = [f"{turn['role']}: {turn['content']}" for turn in chosen_dialog if turn['role'] != 'assistant']
    prompt = "\n".join(prompt_parts)
    
    chosen_response = next(turn["content"] for turn in chosen_dialog if turn["role"] == "assistant")
    rejected_response = next(turn["content"] for turn in rejected_dialog if turn["role"] == "assistant")
    
    examples.append({
        "prompt": prompt,
        "chosen": chosen_response,
        "rejected": rejected_response
    })

# Create Dataset
preference_dataset = Dataset.from_list(examples)
preference_dataset.push_to_hub("alex2020/SimpleDataset-Preference")
