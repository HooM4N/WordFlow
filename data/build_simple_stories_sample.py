from datasets import load_dataset

def sampling_mask(example):
    low, high = word_count_range
    return (
        (example["grammar"] == "") &
        (example["word_count"] >= low) &
        (example["word_count"] <= high) &
        (example["flesch_reading_ease"] >= reading_ease) &
        (example["initial_word_type"] in init_word_types) &
        (example["style"] in styles)
    )

def newline_replace(example):
    return {"story": example["story"].replace("\n\n", " xxspecialxxnewlinexx ")}
    
def build_simple_stories_sample():
    ds = load_dataset(simple_stories_path, split="train")
    ds = ds.filter(sampling_mask)
    ds = ds.map(newline_replace)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(ds["story"]))

if __name__ == "__main__":
    simple_stories_path = "/mnt/d/ML-Files/Datasets/Text/SimpleStories"
    save_path = "data/simple_stories_filtered.txt"
    
    word_count_range = (100,155)
    styles = ["lighthearted", "adventurous", "fable-like", "modern", "classic",
              "playful", "whimsical", "minimalist", "heartwarming", "romantic"]
    reading_ease = 85
    init_word_types = ["adverb", "noun"]
    
    build_simple_stories_sample()