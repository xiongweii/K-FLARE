import os
import json
import tqdm
import numpy as np
from typing import List, Dict, Any
from config import pheme_dataset_path


class PhemeRNDataSetProcessor:
    def __init__(self, dataset_path: str):
        """
        Initialize dataset processor
        :param dataset_path: Root path of PhemeRNDataset
        """
        self.dataset_path = dataset_path
        self.event_dirs = self._get_event_directories()
        self.label_mapping = {
            'rumours': 1,
            'non-rumours': 0
        }

    def _get_event_directories(self) -> List[str]:
        """Get all event directories"""
        pheme_dir = os.path.join(self.dataset_path, 'pheme-rnr-dataset')
        return [os.path.join(pheme_dir, d) for d in os.listdir(pheme_dir)
                if os.path.isdir(os.path.join(pheme_dir, d))]

    def read_source_tweet(self, source_tweet_dir: str) -> Dict[str, Any]:
        """
        Read source tweet information
        :param source_tweet_dir: Source tweet directory path
        :return: Dictionary containing source tweet information
        """
        source_tweet_file = os.path.join(source_tweet_dir, 'source-tweet', f'{os.path.basename(source_tweet_dir)}.json')
        with open(source_tweet_file, 'r', encoding='utf-8') as f:
            tweet = json.load(f)

        # Extract key information
        return {
            'tweet_id': tweet['id_str'],
            'content': tweet['text'],
            'created_at': tweet['created_at'],
            'user_id': tweet['user']['id_str'],
            'user_followers': tweet['user']['followers_count'],
            'user_friends': tweet['user']['friends_count'],
            'user_verified': tweet['user']['verified']
        }

    def read_reactions(self, reactions_dir: str, source_tweet_time: str) -> List[Dict[str, Any]]:
        """
        Read all reply tweets
        :param reactions_dir: Reply folder path
        :param source_tweet_time: Source tweet publish time
        :return: List of reply tweets
        """
        from datetime import datetime
        import dateutil.parser

        reactions = []
        for file_name in os.listdir(reactions_dir):
            if file_name.endswith('.json'):
                file_path = os.path.join(reactions_dir, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    reaction = json.load(f)

                # Calculate time delay
                try:
                    reaction_time = dateutil.parser.parse(reaction['created_at'])
                    source_time = dateutil.parser.parse(source_tweet_time)
                    time_delay = (reaction_time - source_time).total_seconds()  # Keep as seconds
                except Exception as e:
                    time_delay = 0.0

                # Extract key reply information
                reactions.append({
                    'tweet_id': reaction['id_str'],
                    'content': reaction['text'],
                    'created_at': reaction['created_at'],
                    'user_id': reaction['user']['id_str'],
                    'in_reply_to': reaction.get('in_reply_to_status_id_str', None),
                    'time_delay': time_delay,
                    'user_followers': reaction['user']['followers_count'],
                    'user_friends': reaction['user']['friends_count'],
                    'user_verified': reaction['user']['verified']
                })

        return reactions

    def build_propagation_tree(self, source_tweet: Dict[str, Any], reactions: List[Dict[str, Any]]) -> List[
        Dict[str, Any]]:
        """
        Build propagation tree structure, maintaining consistent format with read_tree_file
        :param source_tweet: Source tweet information
        :param reactions: List of reply tweets
        :return: List of propagation tree edges
        """
        tree_edges = []

        # First add edge from root node to source tweet
        root_node = ['ROOT', 'ROOT', '0.0']
        source_node = [
            source_tweet['user_id'],
            source_tweet['tweet_id'],
            '0.0'  # Source tweet time delay is 0
        ]

        tree_edges.append({
            'parent_uid': root_node[0],
            'parent_tid': root_node[1],
            'parent_time': float(root_node[2]),
            'child_uid': source_node[0],
            'child_tid': source_node[1],
            'child_time': float(source_node[2]),
            'time_delay': 0.0
        })

        # Build edges between replies
        for reaction in reactions:
            # Find parent node for reply (default reply to source tweet)
            parent_tid = reaction['in_reply_to'] or source_tweet['tweet_id']
            parent_node = None

            # Search for parent node
            if parent_tid == source_tweet['tweet_id']:
                parent_node = source_node
            else:
                # Search for parent node in already built edges
                for edge in tree_edges:
                    if edge['child_tid'] == parent_tid:
                        parent_node = [edge['child_uid'], edge['child_tid'], str(edge['child_time'])]
                        break

            # If parent node is found, add edge
            if parent_node:
                child_node = [
                    reaction['user_id'],
                    reaction['tweet_id'],
                    str(reaction['time_delay'])  # Keep as string format
                ]

                tree_edges.append({
                    'parent_uid': parent_node[0],
                    'parent_tid': parent_node[1],
                    'parent_time': float(parent_node[2]),
                    'child_uid': child_node[0],
                    'child_tid': child_node[1],
                    'child_time': float(child_node[2]),
                    'time_delay': float(child_node[2]) - float(parent_node[2])
                })

        return tree_edges

    def extract_propagation_features(self, tree_edges: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract features from propagation tree
        :param tree_edges: List of propagation tree edges
        :return: Propagation features dictionary
        """
        # Calculate time features (in hours)
        time_delays = [edge['time_delay'] / 3600 for edge in tree_edges if edge['time_delay'] > 0]  # Convert to hours

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
        for edge in tree_edges:
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

    def process_data(self) -> List[Dict[str, Any]]:
        """
        Process dataset and build data items
        :return: List containing all processed data items
        """
        all_data = []

        # Iterate through each event
        for event_dir in tqdm.tqdm(self.event_dirs, desc="Processing events"):
            event_name = os.path.basename(event_dir)

            # Iterate through rumours and non-rumours folders
            for category in ['rumours', 'non-rumours']:
                category_dir = os.path.join(event_dir, category)
                if not os.path.isdir(category_dir):
                    continue

                # Get all source tweet folders under this category
                tweet_dirs = [os.path.join(category_dir, d) for d in os.listdir(category_dir)
                              if os.path.isdir(os.path.join(category_dir, d))]

                # Process each source tweet
                for tweet_dir in tqdm.tqdm(tweet_dirs, desc=f"Processing {category} in {event_name}", leave=False):
                    tweet_id = os.path.basename(tweet_dir)

                    try:
                        # Read source tweet
                        source_tweet = self.read_source_tweet(tweet_dir)

                        # Read replies
                        reactions_dir = os.path.join(tweet_dir, 'reactions')
                        reactions = self.read_reactions(reactions_dir, source_tweet['created_at'])

                        # Build propagation tree
                        tree_edges = self.build_propagation_tree(source_tweet, reactions)

                        # Extract propagation features
                        propagation_features = self.extract_propagation_features(tree_edges)

                        # Build data item
                        data_item = {
                            'tweet_id': tweet_id,
                            'event': event_name,
                            'label': self.label_mapping[category],
                            'content': source_tweet['content'],
                            'source_user_id': source_tweet['user_id'],
                            'source_user_followers': source_tweet['user_followers'],
                            'source_user_friends': source_tweet['user_friends'],
                            'source_user_verified': source_tweet['user_verified'],
                            'propagation_features': propagation_features,
                            'tree_data': tree_edges,
                            'reactions_count': len(reactions)
                        }

                        all_data.append(data_item)

                    except Exception as e:
                        print(f"Error processing tweet {tweet_id}: {e}")

        return all_data


# Initialize processor
processor = PhemeRNDataSetProcessor(dataset_path=pheme_dataset_path)

# Process data and build data items
data_items = processor.process_data()

# Save results (optional)
with open('pheme_processed_data.json', 'w', encoding='utf-8') as f:
    json.dump(data_items, f, ensure_ascii=False, indent=2)