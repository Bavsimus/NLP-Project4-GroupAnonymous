from datasets import load_dataset

dataset = load_dataset("tatoeba", lang1="en", lang2="tr")
print(dataset)
print(dataset["train"][0])

dataset.save_to_disk("open_subtitles_en_tr")
print("Dataset saved to disk at 'open_subtitles_en_tr'")
