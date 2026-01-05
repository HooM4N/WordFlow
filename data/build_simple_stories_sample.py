from datasets import load_dataset

# simple_stories_path = "/mnt/d/ML-Files/Datasets/Text/SimpleStories"
simple_stories_path = "/mnt/d/SimpleStories/"

save_path = "data/simple_stories_filtered.txt"
    
word_count_range = (120,160)
# styles = ["lighthearted", "adventurous", "fable-like", "modern", "classic",
#               "playful", "whimsical", "minimalist", "heartwarming", "romantic"]
reading_ease = 85
init_word_types = ["adverb", "noun"]

def sampling_mask(example):
    low, high = word_count_range
    return (
        (example["grammar"] == "") &
        (example["word_count"] >= low) &
        (example["word_count"] <= high) &
        (example["flesch_reading_ease"] >= reading_ease) &
        (example["initial_word_type"] in init_word_types)
        # (example["style"] in styles)
    )

def newline_replace(example):
    return {"story": example["story"].replace("\n\n", " xxspecialxxnewlinexx ")}
    
def build_simple_stories_sample():
    ds = load_dataset(simple_stories_path, split="train")
    ds = ds.filter(sampling_mask)
    ds = ds.map(newline_replace)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(ds["story"]))

## PANDAS FILTERING EXPERIMENT ##
# ds = load_dataset(simple_stories_path, split="train")
# df = ds.to_pandas()

# fdf = df[df["grammar"] == ""]
# print(fdf.shape)

# fdf = fdf[fdf["word_count"].between(word_count_range[0], word_count_range[1])]
# print(fdf.shape)

# fdf = fdf[fdf["initial_word_type"].isin(init_word_types)]
# print(fdf.shape)

# fdf = fdf[fdf["flesch_reading_ease"] >= reading_ease]
# print(fdf.shape)

# print(f'words: {fdf["word_count"].sum():,}')

if __name__ == "__main__":
    build_simple_stories_sample()
