from glob import glob
import pandas as pd



def load_subtitles_dataset(dataset_path):
    # Get all files and sort them properly
    subtitles_paths = glob(dataset_path+'/*.ass')
    subtitles_paths = sorted(subtitles_paths, key=lambda x: (
        int(x.split('Season')[1].split('-')[0]),  # Sort by season number
        int(x.split('-')[1].split('.')[0].strip())  # Sort by episode number
    ))

    scripts = []
    episode_num = []

    for path in subtitles_paths:
        # Read Lines
        with open(path,'r') as file:
            lines = file.readlines()
            lines = lines[27:]
            lines = [ ",".join(line.split(',')[9:]) for line in lines ]
        
        lines = [ line.replace('\\N',' ') for line in lines]
        script = " ".join(lines)

        episode = int(path.split('-')[-1].split('.')[0].strip())

        scripts.append(script)
        episode_num.append(episode)

    df = pd.DataFrame.from_dict({"episode": episode_num, "script": scripts})
    # Sort by episode number to ensure systematic ordering
    df = df.sort_values('episode').reset_index(drop=True)
    return df

# Load the dataset
dataset_path = "../data/Subtitles"
df = load_subtitles_dataset(dataset_path)

# Now df.head() will show episodes in order
df.head()