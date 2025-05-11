import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import time
from typing import List, Dict, Tuple
import json
import os
from collections import defaultdict

# Initialize the sentence transformer model for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

class ChatbotEvaluator:
    def __init__(self):
        self.response_times = []
        self.semantic_similarities = []
        self.query_lengths = []
        self.response_lengths = []
        self.successful_queries = 0
        self.failed_queries = 0
        self.total_queries = 0
        self.evaluation_data = defaultdict(list)
        
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using sentence embeddings."""
        try:
            embeddings = model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return 0.0

    def log_interaction(self, query: str, response: str, ground_truth: str = None, 
                       response_time: float = None, success: bool = True):
        """Log a single interaction for evaluation."""
        self.total_queries += 1
        if success:
            self.successful_queries += 1
        else:
            self.failed_queries += 1

        # Store query and response lengths
        self.query_lengths.append(len(query.split()))
        self.response_lengths.append(len(response.split()))

        # Store response time
        if response_time:
            self.response_times.append(response_time)

        # Calculate semantic similarity if ground truth is provided
        if ground_truth:
            similarity = self.calculate_semantic_similarity(response, ground_truth)
            self.semantic_similarities.append(similarity)

        # Store evaluation data
        self.evaluation_data['queries'].append({
            'query': query,
            'response': response,
            'ground_truth': ground_truth,
            'response_time': response_time,
            'success': success
        })

    def get_metrics(self) -> Dict:
        """Calculate and return all evaluation metrics."""
        metrics = {
            'total_queries': self.total_queries,
            'success_rate': (self.successful_queries / self.total_queries * 100) if self.total_queries > 0 else 0,
            'average_response_time': np.mean(self.response_times) if self.response_times else 0,
            'average_semantic_similarity': np.mean(self.semantic_similarities) if self.semantic_similarities else 0,
            'average_query_length': np.mean(self.query_lengths) if self.query_lengths else 0,
            'average_response_length': np.mean(self.response_lengths) if self.response_lengths else 0,
            'response_time_percentiles': {
                '50th': np.percentile(self.response_times, 50) if self.response_times else 0,
                '90th': np.percentile(self.response_times, 90) if self.response_times else 0,
                '95th': np.percentile(self.response_times, 95) if self.response_times else 0
            }
        }
        return metrics

    def generate_report(self) -> str:
        """Generate a detailed evaluation report."""
        metrics = self.get_metrics()
        report = """
Medical Chatbot Evaluation Report
===============================

Performance Metrics:
------------------
Total Queries: {}
Success Rate: {:.2f}%
Average Response Time: {:.3f} seconds

Response Quality Metrics:
----------------------
Average Semantic Similarity: {:.3f}
Average Query Length: {:.1f} words
Average Response Length: {:.1f} words

Response Time Percentiles:
-----------------------
50th percentile: {:.3f} seconds
90th percentile: {:.3f} seconds
95th percentile: {:.3f} seconds
""".format(
            metrics['total_queries'],
            metrics['success_rate'],
            metrics['average_response_time'],
            metrics['average_semantic_similarity'],
            metrics['average_query_length'],
            metrics['average_response_length'],
            metrics['response_time_percentiles']['50th'],
            metrics['response_time_percentiles']['90th'],
            metrics['response_time_percentiles']['95th']
        )
        return report

    def save_evaluation_data(self, filename: str = 'evaluation_results.json'):
        """Save evaluation data to a JSON file."""
        with open(filename, 'w') as f:
            json.dump({
                'metrics': self.get_metrics(),
                'interactions': dict(self.evaluation_data)
            }, f, indent=4)

    def load_evaluation_data(self, filename: str = 'evaluation_results.json'):
        """Load evaluation data from a JSON file."""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
                self.evaluation_data = defaultdict(list, data.get('interactions', {}))
                # Recalculate metrics based on loaded data
                for query_data in self.evaluation_data['queries']:
                    if query_data['success']:
                        self.successful_queries += 1
                    else:
                        self.failed_queries += 1
                    self.total_queries += 1
                    if query_data['response_time']:
                        self.response_times.append(query_data['response_time'])
                    if query_data['ground_truth']:
                        similarity = self.calculate_semantic_similarity(
                            query_data['response'], 
                            query_data['ground_truth']
                        )
                        self.semantic_similarities.append(similarity) 