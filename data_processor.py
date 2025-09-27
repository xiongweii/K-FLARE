import os

import emoji
import numpy as np
import json
import re

from tqdm import tqdm


class DataProcessor:
    def __init__(self, dataset_dir, dataset_name):
        """
        Initialize data processor

        Parameters:
            dataset_dir: Dataset directory
            dataset_name: Dataset name (twitter15 or twitter16)
        """
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.tree_dir = os.path.join(dataset_dir, dataset_name, 'tree')
        self.label_file = os.path.join(dataset_dir, dataset_name, 'label.txt')
        self.source_tweets_file = os.path.join(dataset_dir, dataset_name, 'source_tweets.txt')

        # Only focus on data with true and false labels
        self.valid_labels = ['true', 'false']

    def read_labels(self):
        """Read label data"""
        labels = {}
        with open(self.label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    label, tweet_id = line.split(':')
                    # Only keep data with true and false labels
                    if label.lower() in self.valid_labels:
                        labels[tweet_id.strip()] = 1 if label.lower() == 'true' else 0
        return labels

    def clean_text(self, text):
        """
        Clean text, including removing URLs, @usernames, handling emojis, etc.

        Parameters:
            text: Text to be cleaned

        Returns:
            Cleaned text
        """

        # Replace emojis with text descriptions
        text = emoji.demojize(text)

        # Replace redundant spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    def read_source_tweets(self):
        """Read source tweet content"""
        source_tweets = {}
        with open(self.source_tweets_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        tweet_id, content = parts
                        source_tweets[tweet_id.strip()] = self.clean_text(content)
        return source_tweets



    def read_tree_file(self, file_path):
        """Read propagation tree file"""
        edges = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Parse edge information
                    parent, child = line.split('->')
                    parent_info = eval(parent.strip())
                    child_info = eval(child.strip())

                    # Extract user ID, tweet ID and time delay
                    parent_uid, parent_tid, parent_time = parent_info
                    child_uid, child_tid, child_time = child_info

                    # Build edge information
                    edge = {
                        'parent_uid': parent_uid,
                        'parent_tid': parent_tid,
                        'parent_time': float(parent_time),
                        'child_uid': child_uid,
                        'child_tid': child_tid,
                        'child_time': float(child_time),
                        'time_delay': float(child_time) - float(parent_time)
                    }
                    edges.append(edge)
        return edges

    def extract_propagation_features(self, tree_data):
        """Extract features from propagation tree"""
        # Calculate time features (time delays in tree files are in minutes, convert to hours)
        time_delays = [edge['time_delay'] / 60 for edge in tree_data if edge['time_delay'] > 0]  # Minutes to hours

        if not time_delays:
            # If no valid time delay data, return default values
            return {
                'avg_time_delay': 0,
                'max_time_delay': 0,
                'max_depth': 0,
                'time_delay_feature': [[0.0] * 12]
            }

        avg_time_delay = np.mean(time_delays)
        max_time_delay = max(time_delays)
        min_time_delay = min(time_delays)

        # Limit maximum time to 72 hours (3 days)
        max_time = min(max_time_delay, 72)

        # Handle extreme cases: if only one time point
        if max_time == min_time_delay:
            time_span = 1.0
        else:
            time_span = max_time - min_time_delay

        # Create time window divisions (12 time windows)
        if time_span > 0:
            time_bins = np.linspace(min_time_delay, max_time + 1e-6, 12 + 1)
        else:
            time_bins = np.linspace(0, 12, 13)

        # Count propagation activity in each time window
        hist, _ = np.histogram(time_delays, bins=time_bins)

        # Normalization processing, convert to proportions
        if np.sum(hist) > 0:
            normalized_hist = hist / np.sum(hist)
        else:
            normalized_hist = np.zeros(12)

        time_delays_feature = normalized_hist.reshape(1, -1)

        # Calculate depth
        depths = {}
        for edge in tree_data:
            parent, child = edge['parent_uid'], edge['child_uid']
            if parent == 'ROOT':
                depths[child] = 1
            elif parent in depths:
                depths[child] = depths[parent] + 1

        max_depth = max(depths.values()) if depths else 0

        # Build feature vector
        features = {
            'avg_time_delay': avg_time_delay,
            'max_time_delay': max_time_delay,
            'max_depth': max_depth,
            'time_delay_feature': time_delays_feature.tolist()
        }

        return features

    def process_data(self):
        """Process data and create dataset"""
        # Read labels and source tweets
        labels = self.read_labels()
        source_tweets = self.read_source_tweets()

        # Filter to include only data with true and false labels
        valid_tweet_ids = set(labels.keys())

        data = []

        # Process each propagation tree
        for tweet_id in tqdm(valid_tweet_ids):
            if tweet_id in source_tweets:
                tree_file = os.path.join(self.tree_dir, f'{tweet_id}.txt')

                if os.path.exists(tree_file):
                    # Read propagation tree
                    tree_data = self.read_tree_file(tree_file)

                    # Extract propagation features
                    propagation_features = self.extract_propagation_features(tree_data)

                    # Create data item
                    item = {
                        'tweet_id': tweet_id,
                        'label': labels[tweet_id],
                        'content': source_tweets[tweet_id],
                        'propagation_features': propagation_features,
                        'tree_data': tree_data
                    }

                    data.append(item)

        return data

    def save_processed_data(self, output_file):
        """Save processed data"""
        data = self.process_data()
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved processed data to {output_file}, total {len(data)} records")
        return data
